"""TabTransformer model"""

import logging
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from autogluon.common.features.types import R_OBJECT, S_TEXT_AS_CATEGORY, S_TEXT_NGRAM
from autogluon.common.utils.try_import import try_import_torch
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models.abstract.abstract_nn_model import AbstractNeuralNetworkModel
from autogluon.core.utils.loaders import load_pkl

from .hyperparameters.parameters import get_default_param
from .hyperparameters.searchspaces import get_default_searchspace

logger = logging.getLogger(__name__)

"""
TODO: Fix Mac OS X warning spam.
The error message is:
Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)
This has been investigated to be a harmless warning for training and running inference on TabTransformer.
This warning can occur with a very specific environment: torch 1.7, Mac OS X, Python 3.6/3.7, when using torch DataLoader.
https://github.com/pytorch/pytorch/issues/46409
"""


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
        self._verbosity = None
        self._temp_file_name = "tab_trans_temp.pth"
        self._period_columns_mapping = None

    def _set_default_params(self):
        default_params = get_default_param()
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[R_OBJECT],
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_AS_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _get_model(self):
        from .tab_model_base import TabNet

        # If we have already initialized the model, we don't need to do it again.
        model = TabNet(self.params["n_classes"], self.params["feature_dim"], self.params["num_output_layers"], self.device, self.params)
        if self.device.type == "cuda":
            model = model.cuda()

        return model

    # NOTE: Making an assumption that X_unlabeled will not have a different schema. Otherwise, we would need two
    # period_columns_mapping fields. One for X/X_val, another for X_unlabeled, which may have different columns.
    @staticmethod
    def _get_no_period_columns(columns):
        # Latest pytorch does not support . in module names. Therefore, we must replace the ".".
        rename_columns = dict()
        for col in columns:
            new_col_name = col
            if "." in col:
                new_col_name = col.replace(".", "_")

            if new_col_name in rename_columns:
                for i in range(1, 100):
                    append_col_name = new_col_name + "_" + str(i)
                    if append_col_name not in rename_columns:
                        new_col_name = append_col_name
                        break
                else:
                    raise RuntimeError("Tried 100 column renames to eliminate duplicates.\n" "Please check similar columns with . or _ in them.")

            # Mapping for every column
            rename_columns[col] = new_col_name

        return rename_columns

    def _preprocess(self, X, **kwargs):
        from .utils import TabTransformerDataset

        X = super()._preprocess(X=X, **kwargs)

        X = X.rename(columns=self._period_columns_mapping)
        encoders = self.params["encoders"]
        data = TabTransformerDataset(X, encoders=encoders, problem_type=self.problem_type, col_info=self._types_of_features)
        data.encode(self.fe)

        return data

    def _preprocess_train(self, X, X_val=None, X_unlabeled=None, fe=None):
        """
        Pre-processing specific to TabTransformer. Setting up feature encoders, renaming columns with periods in
        them (torch), and converting X, X_val, X_unlabeled into TabTransformerDataset's.
        """
        from .utils import TabTransformerDataset

        X = self._preprocess_nonadaptive(X)
        if X_val is not None:
            X_val = self._preprocess_nonadaptive(X_val)
        if X_unlabeled is not None:
            X_unlabeled = self._preprocess_nonadaptive(X_unlabeled)

        self._period_columns_mapping = self._get_no_period_columns(X.columns)
        X = X.rename(columns=self._period_columns_mapping)

        if X_val is not None:
            X_val = X_val.rename(columns=self._period_columns_mapping)
        if X_unlabeled is not None:
            X_unlabeled = X_unlabeled.rename(columns=self._period_columns_mapping)

        self._types_of_features, _ = self._get_types_of_features(X, needs_extra_types=False)

        # Also need to rename the feature names in the types_of_features dictionary.
        for feature_dict in self._types_of_features:
            # Need to check that the value is in the mapping. Otherwise, we could be updating columns that have been dropped.
            feature_dict.update(("name", self._period_columns_mapping[v]) for k, v in feature_dict.items() if k == "name" and v in self._period_columns_mapping)

        encoders = self.params["encoders"]
        data = TabTransformerDataset(X, encoders=encoders, problem_type=self.problem_type, col_info=self._types_of_features)
        self.fe = fe
        if self.fe is not None:
            if X_unlabeled is None:
                unlab_data = None
            elif X_unlabeled is not None:
                unlab_data = TabTransformerDataset(X_unlabeled, encoders=encoders, problem_type=self.problem_type, col_info=self._types_of_features)
        if self.fe is None:
            if X_unlabeled is None:
                data.fit_feat_encoders()
                self.fe = data.feature_encoders
                unlab_data = None
            elif X_unlabeled is not None:
                unlab_data = TabTransformerDataset(X_unlabeled, encoders=encoders, problem_type=self.problem_type, col_info=self._types_of_features)
                unlab_data.fit_feat_encoders()
                self.fe = unlab_data.feature_encoders

        data.encode(self.fe)

        if X_val is not None:
            val_data = TabTransformerDataset(X_val, encoders=encoders, problem_type=self.problem_type, col_info=self._types_of_features)
            val_data.encode(self.fe)
        else:
            val_data = None

        if unlab_data is not None:
            unlab_data.encode(self.fe)

        return data, val_data, unlab_data

    def _epoch(
        self, net, loader_train, loader_val, y_val, optimizers, loss_criterion, pretext, state, scheduler, epoch, epochs, databar_disable, reporter, params
    ):
        """
        Helper function to run one epoch of training, essentially the "inner loop" of training.
        """
        import torch

        from .utils import augmentation

        is_train = optimizers is not None
        net.train() if is_train else net.eval()
        total_loss, total_correct, total_num = 0.0, 0.0, 0
        data_bar = tqdm(loader_train, disable=databar_disable) if is_train else tqdm(loader_val, disable=databar_disable)

        with torch.enable_grad() if is_train else torch.no_grad():
            for data, target in data_bar:
                data, target = pretext.get(data, target)

                if self.device.type == "cuda":
                    data, target = data.cuda(), target.cuda()
                    pretext = pretext.cuda()

                if state in [None, "finetune"]:
                    if self.params["num_augs"] > 0:
                        data, target = augmentation(data, target, **params)
                    out, _ = net(data)
                elif state == "pretrain":
                    _, out = net(data)
                else:
                    raise NotImplementedError("state must be one of [None, 'pretrain', 'finetune']")

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
                    train_test = "Test"
                else:
                    train_test = "Train"

                val_metric = None
                if loader_val is not None and state != "pretrain":
                    val_metric = self.score(X=loader_val, y=y_val, metric=self.stopping_metric)
                    data_bar.set_description(
                        "{} Epoch: [{}/{}] Train Loss: {:.4f} Validation {}: {:.2f}".format(
                            train_test, epoch, epochs, total_loss / total_num, self.stopping_metric.name, val_metric
                        )
                    )

                    if reporter is not None:
                        reporter(epoch=epoch + 1, validation_performance=val_metric, train_loss=total_loss)

                else:
                    data_bar.set_description("{} Epoch: [{}/{}] Loss: {:.4f}".format(train_test, epoch, epochs, total_loss / total_num))

            return total_loss / total_num, val_metric

        if scheduler is not None:
            scheduler.step()
        return total_loss / total_num

    def tt_fit(self, loader_train, loader_val=None, y_val=None, state=None, time_limit=None, reporter=None):
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
        lr = self.params["lr"]
        weight_decay = self.params["weight_decay"]
        epochs = self.params["pretrain_epochs"] if state == "pretrain" else self.params["epochs"]
        epochs_wo_improve = self.params["epochs_wo_improve"]

        if state is None:
            optimizers = [optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)]
            pretext = pretext_tasks["SupervisedPretext"](self.problem_type, self.device)
        elif state == "pretrain":
            optimizers = [optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)]
            pretext = pretext_tasks["BERTPretext"](self.cat_feat_origin_cards, self.device, self.params["hidden_dim"])
        elif state == "finetune":
            base_exp_decay = self.params["base_exp_decay"]
            optimizer_fc = [optim.Adam(fc_layer.parameters(), lr=lr, weight_decay=weight_decay) for fc_layer in self.model.fc]
            optimizer_embeds = optim.Adam(self.model.embed.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer_embeds, gamma=base_exp_decay)  # TODO: Should we be using this in _epoch()?
            optimizers.extend(optimizer_fc)
            optimizers.append(optimizer_embeds)

            pretext = pretext_tasks["SupervisedPretext"](self.problem_type, self.device)

        else:
            raise NotImplementedError("state must be one of [None, 'pretrain', 'finetune']")

        if self.problem_type == REGRESSION:
            loss_criterion = nn.MSELoss()
        else:
            loss_criterion = nn.CrossEntropyLoss()

        best_val_metric = -np.inf  # higher = better
        best_val_epoch = 0
        best_loss = np.inf

        if self._verbosity <= 1:
            verbose_eval = -1
        elif self._verbosity == 2:
            verbose_eval = 50
        elif self._verbosity == 3:
            verbose_eval = 10
        else:
            verbose_eval = 1

        if verbose_eval <= 0:
            databar_disable = True  # Whether or not we want to suppress output based on our verbosity
        else:
            databar_disable = False

        for e in range(epochs):
            if e == 0:
                logger.log(15, "TabTransformer architecture:")
                logger.log(15, str(self.model))

            train_loss, val_metric = self._epoch(
                net=self.model,
                loader_train=loader_train,
                loader_val=loader_val,
                y_val=y_val,
                optimizers=optimizers,
                loss_criterion=loss_criterion,
                pretext=pretext,
                state=state,
                scheduler=None,
                epoch=e,
                epochs=epochs,
                databar_disable=databar_disable,
                reporter=reporter,
                params=self.params,
            )

            # Early stopping for pretrain'ing based on loss.
            if state == "pretrain":
                if train_loss < best_loss or e == 0:
                    if train_loss < best_loss:
                        best_loss = train_loss
                    best_val_epoch = e
            else:
                if val_metric >= best_val_metric or e == 0:
                    if loader_val is not None:
                        if not np.isnan(val_metric):
                            best_val_metric = val_metric

                    best_val_epoch = e
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                    torch.save(self.model, os.path.join(self.path, self._temp_file_name)) # nosec B614

            # If time limit has exceeded or we haven't improved in some number of epochs, stop early.
            if e - best_val_epoch > epochs_wo_improve:
                break
            if time_limit:
                time_elapsed = time.time() - start_time
                time_left = time_limit - time_elapsed
                if time_left <= 0:
                    logger.log(20, "\tRan out of time, stopping training early.")
                    break

        if loader_val is not None:
            try:
                self.model = torch.load(os.path.join(self.path, self._temp_file_name)) # nosec B614
                os.remove(os.path.join(self.path, self._temp_file_name))
            except:
                pass
            logger.log(15, "Best model found in epoch %d" % best_val_epoch)

    def _fit(self, X, y, X_val=None, y_val=None, X_unlabeled=None, time_limit=None, sample_weight=None, reporter=None, **kwargs):
        import torch

        self._verbosity = kwargs.get("verbosity", 2)
        num_gpus = kwargs.get("num_gpus", None)
        if num_gpus is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        elif num_gpus == 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

            if num_gpus > 1:
                logger.warning("TabTransformer not yet configured to use more than 1 GPU. 'num_gpus' set to >1, but we will be using only 1 GPU.")

        if sample_weight is not None:
            logger.log(15, "sample_weight not yet supported for TabTransformerModel, this model will ignore them in training.")

        if self.problem_type == REGRESSION:
            self.params["n_classes"] = 1
        elif self.problem_type == BINARY:
            self.params["n_classes"] = 2
        elif self.problem_type == MULTICLASS:
            self.params["n_classes"] = y.nunique()

        train, val, unlab = self._preprocess_train(X, X_val, X_unlabeled)

        num_cols = len(train.columns)
        if num_cols > self.params["max_columns"]:
            raise NotImplementedError(
                f"This dataset has {num_cols} columns and exceeds 'max_columns' == {self.params['max_columns']}.\n"
                f"Which is set by default to ensure the TabTransformer model will not run out of memory.\n"
                f"If you are confident you will have enough memory, set the 'max_columns' hyperparameter higher and try again.\n"
            )

        if self.problem_type == REGRESSION:
            train.targets = torch.FloatTensor(list(y))
            val.targets = torch.FloatTensor(list(y_val))
        else:
            train.targets = torch.LongTensor(list(y))
            val.targets = torch.LongTensor(list(y_val))

        batch_size = self.params["batch_size"]
        num_workers = self.params["num_workers"]

        loader_train = train.build_loader(batch_size, num_workers, shuffle=True)
        loader_val = val.build_loader(batch_size, num_workers)
        loader_unlab = unlab.build_loader(batch_size, num_workers) if unlab is not None else None

        self.cat_feat_origin_cards = loader_train.cat_feat_origin_cards
        self.params["cat_feat_origin_cards"] = self.cat_feat_origin_cards

        self.model = self._get_model()

        if X_unlabeled is not None:
            # Can't spend all the time in pretraining, have to split it up.
            pretrain_time_limit = time_limit / 2 if time_limit is not None else time_limit
            pretrain_before_time = time.time()
            self.tt_fit(loader_unlab, loader_val, y_val, state="pretrain", time_limit=pretrain_time_limit, reporter=reporter)
            finetune_time_limit = time_limit - (time.time() - pretrain_before_time) if time_limit is not None else time_limit
            self.tt_fit(loader_train, loader_val, y_val, state="finetune", time_limit=finetune_time_limit, reporter=reporter)
        else:
            self.tt_fit(loader_train, loader_val, y_val, time_limit=time_limit, reporter=reporter)

    def _predict_proba(self, X, **kwargs):
        """
        X (torch.tensor or pd.dataframe): data for model to give prediction probabilities
        returns: np.array of k-probabilities for each of the k classes. If k=2 we drop the second probability.
        """
        import torch
        import torch.nn as nn
        from torch.autograd import Variable
        from torch.utils.data import DataLoader

        if isinstance(X, pd.DataFrame):
            # Preprocess here also calls our _preprocess, which creates a TTDataset.
            X = self.preprocess(X, **kwargs)
            loader = X.build_loader(self.params["batch_size"], self.params["num_workers"])
        elif isinstance(X, DataLoader):
            loader = X
        elif isinstance(X, torch.Tensor):
            X = X.rename(columns=self._get_no_period_columns(X))
            loader = X.build_loader(self.params["batch_size"], self.params["num_workers"])
        else:
            raise NotImplementedError(
                "Attempting to predict against a non-supported data type. \nNeeds to be a pandas DataFrame, torch DataLoader or torch Tensor."
            )

        self.model.eval()
        softmax = nn.Softmax(dim=1)

        if self.problem_type == REGRESSION:
            outputs = torch.zeros([len(loader.dataset), 1])
        else:
            outputs = torch.zeros([len(loader.dataset), self.num_classes])

        iter = 0
        for data, _ in loader:
            if self.device.type == "cuda":
                data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
                prob, _ = self.model(data)
                batch_size = len(prob)
                if self.problem_type != REGRESSION:
                    prob = softmax(prob)

            outputs[iter : (iter + batch_size)] = prob
            iter += batch_size

        if self.problem_type == BINARY:
            return outputs[:, 1].cpu().numpy()
        elif self.problem_type == REGRESSION:
            outputs = outputs.flatten()

        return outputs.cpu().numpy()

    def _get_default_searchspace(self):
        return get_default_searchspace()

    def save(self, path: str = None, verbose=True) -> str:
        import torch

        if path is None:
            path = self.path

        params_filepath = os.path.join(path, self.params_file_name)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        temp_model = self.model
        if self.model is not None:
            torch.save(self.model, params_filepath) # nosec B614

        self.model = None  # Avoiding pickling the weights.
        modelobj_filepath = super().save(path=path, verbose=verbose)

        self.model = temp_model

        return modelobj_filepath

    @classmethod
    def load(cls, path: str, reset_paths=False, verbose=True):
        import torch

        obj: TabTransformerModel = load_pkl.load(path=os.path.join(path, cls.model_file_name), verbose=verbose)
        if reset_paths:
            obj.set_contexts(path)

        obj.model = torch.load(os.path.join(path, cls.params_file_name)) # nosec B614

        return obj

        """
        List of features to add (Updated by Anthony Galczak 11-19-20):

        1) Allow for saving of pretrained model for future use. This will be done in a future PR as the
        "pretrain API change".

        2) Investigate options for when the unlabeled schema does not match the training schema. Currently,
        we do not allow such mismatches and the schemas must match exactly. We can investigate ways to use
        less or more columns from the unlabeled data. This will likely require a design meeting.

        3) Bug where HPO doesn't work when cuda is enabled.
        "RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method"
        Update: This will likely be fixed in a future change to HPO in AutoGluon.
        """
