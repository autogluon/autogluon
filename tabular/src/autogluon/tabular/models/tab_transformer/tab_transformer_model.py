""" TabTransformer model """
import logging
import os
import time

import numpy as np
import pandas as pd
from autogluon.core.utils.loaders import load_pkl
from tqdm import tqdm

from .hyperparameters.parameters import get_default_param
from .hyperparameters.searchspaces import get_default_searchspace
from ..abstract.abstract_model import AbstractNeuralNetworkModel
from ...constants import BINARY, REGRESSION, MULTICLASS
from autogluon.core.utils import try_import_torch

logger = logging.getLogger(__name__)


class TabTransformerModel(AbstractNeuralNetworkModel):
    """
    Main TabTransformer model that inherits from AbstractModel.

    This model includes the full torch pipeline (TabNet) and the internal Transformer embeddings (TabTransformer).
    This file serves as the connection of all these internal models and architectures to AutoGluon.

    TabTransformer uses modifications to the typical Transformer architecture and the pretraining in BERT
    and applies them to the use case of tabular data. Specifically, this makes TabTransformer suitable for unsupervised
    training of Tabular data with a subsequent fine-tuning step on labeled data.
    """
    params_file_name = "tab_trans_params.pth"

    def __init__(self, **kwargs):
        try_import_torch()
        super().__init__(**kwargs)
        self.types_of_features = None
        self.verbosity = None
        self.temp_file_name = "tab_trans_temp.pth"

    def _set_default_params(self):
        import torch

        default_params = get_default_param(self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

        # TODO: Take in num_gpu's as a param. Currently this is hard-coded upon detection of cuda.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.params['device'] = device

    def get_model(self):
        from .tab_model_base import TabNet
        # If we have already initialized the model, we don't need to do it again.
        if self.model is None:
            self.model = TabNet(self.params['n_classes'], self.params, self.cat_feat_origin_cards)
            if self.params['device'].type == "cuda":
                self.model = self.model.cuda()

    # TODO: Ensure column name uniqueness. Potential conflict if input column name has "/-#" in it.
    def _get_no_period_columns(self, X):
        # Latest pytorch does not support . in module names. Therefore, we must replace the . with some other symbol
        # that hopefully doesn't collide with other column names.
        rename_columns = dict()
        for col in X.columns:
            if "." in col:
                new_col_name = col.replace(".", "/-#")
                rename_columns[col] = new_col_name

        return rename_columns

    def _tt_preprocess(self, X, X_val=None, X_unlabeled=None, fe=None):
        """
        Pre-processing specific to TabTransformer. Setting up feature encoders, renaming columns with periods in
        them (torch), and converting X, X_val, X_unlabeled into TabTransformerDataset's.
        """
        from .utils import TabTransformerDataset
        X = X.rename(columns=self._get_no_period_columns(X))

        if X_val is not None:
            X_val = X_val.rename(columns=self._get_no_period_columns(X_val))
        if X_unlabeled is not None:
            X_unlabeled = X_unlabeled.rename(columns=self._get_no_period_columns(X_unlabeled))

        self.types_of_features, _ = self._get_types_of_features(X, needs_torch=True, needs_extra_types=False)

        data = TabTransformerDataset(X, col_info=self.types_of_features, **self.params)
        self.fe = fe
        if self.fe is not None:
            if X_unlabeled is None:
                unlab_data = None
            elif X_unlabeled is not None:
                unlab_data = TabTransformerDataset(X_unlabeled, col_info=self.types_of_features, **self.params)
        if self.fe is None:
            if X_unlabeled is None:
                data.fit_feat_encoders()
                self.fe = data.feature_encoders
                unlab_data = None
            elif X_unlabeled is not None:
                unlab_data = TabTransformerDataset(X_unlabeled, col_info=self.types_of_features, **self.params)
                unlab_data.fit_feat_encoders()
                self.fe = unlab_data.feature_encoders

        data.encode(self.fe)

        if X_val is not None:
            val_data = TabTransformerDataset(X_val, col_info=self.types_of_features, **self.params)
            val_data.encode(self.fe)
        else:
            val_data = None

        if unlab_data is not None:
            unlab_data.encode(self.fe)

        return data, val_data, unlab_data

    def _epoch(self, net, trainloader, valloader, y_val, optimizers, loss_criterion, pretext, state, scheduler, epoch,
               epochs, databar_disable, reporter, params):
        """
        Helper function to run one epoch of training, essentially the "inner loop" of training.
        """
        import torch
        from .utils import augmentation
        is_train = (optimizers is not None)
        net.train() if is_train else net.eval()
        total_loss, total_correct, total_num = 0.0, 0.0, 0
        data_bar = tqdm(trainloader, disable=databar_disable) if is_train else tqdm(valloader, disable=databar_disable)

        with (torch.enable_grad() if is_train else torch.no_grad()):
            for data, target in data_bar:
                data, target = pretext.get(data, target)

                if params['device'].type == "cuda":
                    data, target = data.cuda(), target.cuda()
                    pretext = pretext.cuda()

                if state in [None, 'finetune']:
                    data, target = augmentation(data, target, **params)
                    out, _ = net(data)
                elif state == 'pretrain':
                    _, out = net(data)
                else:
                    raise NotImplementedError("state must be one of [None, 'pretrain', 'finetune']")

                # TODO: Is this the right spot to put "out" into GPU? Likely not... memory access error..
                #if params['device'].type == "cuda":
                #    out = out.cuda()

                loss, correct = pretext(out, target)

                if is_train:
                    for optimizer in optimizers:
                        optimizer.zero_grad()
                    loss.backward()
                    for optimizer in optimizers:
                        optimizer.step()

                total_num += 1
                total_loss += loss.item()

                if epochs == 1:
                    train_test = 'Test'
                else:
                    train_test = 'Train'

                val_metric = None
                if valloader is not None and state != 'pretrain':
                    val_metric = self.score(X=valloader, y=y_val, eval_metric=self.stopping_metric,
                                            metric_needs_y_pred=self.stopping_metric_needs_y_pred)
                    data_bar.set_description('{} Epoch: [{}/{}] Train Loss: {:.4f} Validation {}: {:.2f}'.format(
                        train_test, epoch, epochs, total_loss / total_num, self.eval_metric.name, val_metric))

                    if reporter is not None:
                        reporter(epoch=epoch+1, validation_performance=val_metric, train_loss=total_loss)

                else:
                    data_bar.set_description(
                        '{} Epoch: [{}/{}] Loss: {:.4f}'.format(train_test, epoch, epochs, total_loss / total_num))

            return total_loss / total_num, val_metric

        if scheduler is not None:
            scheduler.step()
        return total_loss / total_num

    def tt_fit(self, trainloader, valloader=None, y_val=None, state=None, time_limit=None, reporter=None):
        """
        Main training function for TabTransformer
        "state" must be one of [None, 'pretrain', 'finetune']
        None: corresponds to purely supervised learning
        pretrain: discriminative task will be a pretext task
        finetune: same as supervised learning except that the model base has
                  exponentially decaying learning rate.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from . import pretexts

        start_time = time.time()
        pretext_tasks = pretexts.__dict__
        optimizers = []
        lr = self.params['lr']
        weight_decay = self.params['weight_decay']
        epochs = self.params['pretrain_epochs'] if state == 'pretrain' else self.params['epochs']
        epochs_wo_improve = self.params['epochs_wo_improve']

        if state is None:
            optimizers = [optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)]
            pretext = pretext_tasks['SUPERVISED_pretext'](self.params)
        elif state == 'pretrain':
            optimizers = [optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)]
            pretext = pretext_tasks['BERT_pretext'](self.params)
        elif state == 'finetune':
            base_exp_decay = self.params['base_exp_decay']
            optimizer_fc = optim.Adam(self.model.fc.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer_embeds = optim.Adam(self.model.embed.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer_embeds, gamma=base_exp_decay) # TODO: Should we be using this in _epoch()?
            optimizers.append(optimizer_fc)
            optimizers.append(optimizer_embeds)

            pretext = pretext_tasks['SUPERVISED_pretext'](self.params)

        else:
            raise NotImplementedError("state must be one of [None, 'pretrain', 'finetune']")

        if self.params['problem_type'] == REGRESSION:
            loss_criterion = nn.MSELoss()
        else:
            loss_criterion = nn.CrossEntropyLoss()

        best_val_metric = -np.inf  # higher = better
        best_val_epoch = 0
        best_loss = np.inf

        self.verbosity = self.params.get('verbosity', 2)
        if self.verbosity <= 1:
            verbose_eval = -1
        elif self.verbosity == 2:
            verbose_eval = 50
        elif self.verbosity == 3:
            verbose_eval = 10
        else:
            verbose_eval = 1

        for e in range(epochs):
            if e == 0:
                logger.log(15, "TabTransformer architecture:")
                logger.log(15, str(self.model))

            # Whether or not we want to suppress output based on our verbosity.
            databar_disable = False if e % verbose_eval == 0 else True

            train_loss, val_metric = self._epoch(net=self.model, trainloader=trainloader, valloader=valloader, y_val=y_val,
                                                 optimizers=optimizers, loss_criterion=loss_criterion, \
                                                 pretext=pretext, state=state, scheduler=None, epoch=e, epochs=epochs,
                                                 databar_disable=databar_disable, reporter=reporter, params=self.params)

            # Early stopping for pretrain'ing based on loss.
            if state == 'pretrain':
                if train_loss < best_loss or e == 0:
                    if train_loss < best_loss:
                        best_loss = train_loss
                    best_val_epoch = e
            else:
                if val_metric >= best_val_metric or e == 0:
                    if valloader is not None:
                        if not np.isnan(val_metric):
                            best_val_metric = val_metric

                    best_val_epoch = e
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                    torch.save(self.model, self.path + self.temp_file_name)

            # If time limit has exceeded or we haven't improved in some number of epochs, stop early.
            if e - best_val_epoch > epochs_wo_improve:
                break
            if time_limit:
                time_elapsed = time.time() - start_time
                time_left = time_limit - time_elapsed
                if time_left <= 0:
                    logger.log(20, "\tRan out of time, stopping training early.")
                    break

        if valloader is not None:
            try:
                self.model = torch.load(self.path + self.temp_file_name)
                os.remove(self.path + self.temp_file_name)
            except:
                pass
            logger.log(15, "Best model found in epoch %d" % best_val_epoch)

    def _fit(self, X_train, y_train, X_val=None, y_val=None, X_unlabeled=None, time_limit=None, reporter=None, **kwargs):
        import torch

        if self.params['problem_type']==REGRESSION:
            self.params['n_classes'] = 1
        elif self.params['problem_type']==BINARY:
            self.params['n_classes'] = 2
        elif self.params['problem_type']==MULTICLASS:
            self.params['n_classes'] = y_train.nunique()

        num_cols = X_train.shape[1]
        if num_cols > self.params['max_columns']:
            raise NotImplementedError(
                f"This dataset has {num_cols} columns and exceeds 'max_columns' == {self.params['max_columns']}.\n"
                f"Which is set by default to ensure the TabTransformer model will not run out of memory.\n"
                f"If you are confident you will have enough memory, set the 'max_columns' hyperparameter higher and try again.\n")

        train, val, unlab = self._tt_preprocess(X_train, X_val, X_unlabeled)

        if self.problem_type == REGRESSION:
            train.targets = torch.FloatTensor(list(y_train))
            val.targets = torch.FloatTensor(list(y_val))
        else:
            train.targets = torch.LongTensor(list(y_train))
            val.targets = torch.LongTensor(list(y_val))

        trainloader, valloader, unlabloader = train.build_loader(
            shuffle=True), val.build_loader(), unlab.build_loader() if unlab is not None else None

        self.cat_feat_origin_cards = trainloader.cat_feat_origin_cards

        self.get_model()

        if X_unlabeled is not None:
            self.tt_fit(unlabloader, valloader, y_val, state='pretrain', time_limit=time_limit, reporter=reporter)
            self.tt_fit(trainloader, valloader, y_val, state='finetune', time_limit=time_limit, reporter=reporter)
        else:
            self.tt_fit(trainloader, valloader, y_val, time_limit=time_limit, reporter=reporter)

    def _predict_proba(self, X, **kwargs):
        """
        X (torch.tensor or pd.dataframe): data for model to give prediction probabilities
        returns: np.array of k-probabilities for each of the k classes. If k=2 we drop the second probability.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torch.autograd import Variable

        if isinstance(X, pd.DataFrame):
            X = self.preprocess(X, **kwargs)
            X, _, _ = self._tt_preprocess(X, fe=self.fe)
            loader = X.build_loader()
        elif isinstance(X, DataLoader):
            loader = X
        elif isinstance(X, torch.Tensor):
            X = X.rename(columns=self._get_no_period_columns(X))
            loader = X.build_loader()
        else:
            raise NotImplementedError(
                "Attempting to predict against a non-supported data type. \nNeeds to be a pandas DataFrame, torch DataLoader or torch Tensor.")

        self.model.eval()
        softmax = nn.Softmax(dim=1)

        if self.problem_type == REGRESSION:
            outputs = torch.zeros([len(loader.dataset), 1])
        else:
            outputs = torch.zeros([len(loader.dataset), self.num_classes])

        iter = 0
        for data, _ in loader:
            if self.params['device'].type == "cuda":
                data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
                prob, _ = self.model(data)
                batch_size = len(prob)
                if self.problem_type != REGRESSION:
                    prob = softmax(prob)

            outputs[iter:(iter + batch_size)] = prob
            iter += batch_size

        if self.problem_type == BINARY:
            return outputs[:, 1].cpu().numpy()
        elif self.problem_type == REGRESSION:
            outputs = outputs.flatten()

        return outputs.cpu().numpy()

    def _get_default_searchspace(self):
        return get_default_searchspace()

    # TODO: Consider HPO for pretraining with unlabeled data. (Potential future work)
    # TODO: Does not work correctly when cuda is enabled.
    def hyperparameter_tune(self, X_train, y_train, X_val, y_val, scheduler_options, **kwargs):
        from .utils import tt_trial
        import torch

        time_start = time.time()
        self._set_default_searchspace()
        scheduler_func = scheduler_options[0]
        scheduler_options = scheduler_options[1]

        if scheduler_func is None or scheduler_options is None:
            raise ValueError("scheduler_func and scheduler_options cannot be None for hyperparameter tuning")

        util_args = dict(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model=self,
            time_start=time_start,
            time_limit=scheduler_options['time_out']
        )

        params_copy = self.params.copy()
        tt_trial.register_args(util_args=util_args, **params_copy)

        scheduler = scheduler_func(tt_trial, **scheduler_options)
        scheduler.run()
        scheduler.join_jobs()
        self.model = self.model.to(torch.device("cpu"))
        scheduler.get_training_curves(plot=False, use_legend=False)

        return self._get_hpo_results(scheduler=scheduler, scheduler_options=scheduler_options, time_start=time_start)

    def save(self, file_prefix="", directory=None, return_filename=False, verbose=True):
        """
        file_prefix (str): Appended to beginning of file-name (does not affect directory in file-path).
        directory (str): if unspecified, use self.path as directory
        return_filename (bool): return the file-name corresponding to this save
        """
        import torch
        if directory is not None:
            path = directory + file_prefix
        else:
            path = self.path + file_prefix

        params_filepath = path + self.params_file_name

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.model is not None:
            torch.save(self.model, params_filepath)

        self.model = None  # Avoiding pickling the weights.
        modelobj_filepath = super().save(path=path, verbose=verbose)

        if return_filename:
            return modelobj_filepath

    @classmethod
    def load(cls, path, file_prefix="", reset_paths=False, verbose=True):
        """
        file_prefix (str): Appended to beginning of file-name.
        If you want to load files with given prefix, can also pass arg: path = directory+file_prefix
        """
        import torch
        path = path + file_prefix
        obj: TabTransformerModel = load_pkl.load(path=path + cls.model_file_name, verbose=verbose)
        if reset_paths:
            obj.set_contexts(path)

        obj.model = torch.load(path + cls.params_file_name)

        return obj

        """
        List of features to add (Updated by Anthony Galczak 10-30-20):
        
        1) Allow for saving of pretrained model for future use. This will be done in a future PR as the 
        "pretrain API change".
        
        2) Investigate options for when the unlabeled schema does not match the training schema. Currently,
        we do not allow such mismatches and the schemas must match exactly. We can investigate ways to use
        less or more columns from the unlabeled data. This will likely require a design meeting.
        
        3) Bug where HPO doesn't work when cuda is enabled.
        "RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method"
        
        4) Enable output layer of TT model to be multiple fully connected layers rather than just a single
        linear layer. "TabTransformer2 changes"
        NOTE: This is "partially done" right now as there is two FC layers at the end of TabNet. The main addition
        still needed is to process and concatenate continous features.
        """
