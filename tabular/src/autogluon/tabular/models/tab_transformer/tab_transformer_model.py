""" TabTransformer model """
import time, logging

from autogluon.core.utils.loaders import load_pkl

from ...try_import import try_import_torch

from tqdm import tqdm

from ...constants import BINARY, REGRESSION
from .hyperparameters.parameters import get_default_param

import pandas as pd
import os
import numpy as np

from ..abstract.abstract_model import AbstractModel

logger = logging.getLogger(__name__)

class TabNetClass:
    import torch.nn as nn

    class TabNet(nn.Module):
        def __init__(self, num_class, params, cat_feat_origin_cards):
            super(TabNetClass.TabNet, self).__init__()
            import torch.nn as nn
            from .tab_transformer import TabTransformer
            self.params = params
            self.params['cat_feat_origin_cards']=cat_feat_origin_cards
            self.embed=TabTransformer(**self.params)

            relu, lin = nn.ReLU(), nn.Linear(2*self.params['feature_dim'] , num_class, bias=True)
            self.fc = nn.Sequential(*[relu,lin])

        def forward(self, data):
            features = self.embed(data)
            out = features.mean(dim=1)
            out = self.fc(out)
            return out, features

            
class TabTransformerModel(AbstractModel):
    params_file_name="tab_trans_params.pth"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.types_of_features=None
        self.verbosity = None

        # TODO: Put imports in-line where I need them.
        try_import_torch()


    def _get_types_of_features(self, df):
        """ Returns dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = ordered list of feature-names falling into each category.
            Each value is a list of feature-names corresponding to columns in original dataframe.
            TODO: ensure features with zero variance have already been removed before this function is called.
        """
        if self.types_of_features is not None:
            Warning("Attempting to _get_types_of_features for TabTransformerModel, but previously already did this.")

        feature_types = self.feature_metadata.get_type_group_map_raw()

        categorical_featnames = feature_types['category'] + feature_types['object'] + feature_types['bool']
        continuous_featnames = feature_types['float'] + feature_types['int']  # + self.__get_feature_type_if_present('datetime')
        language_featnames = [] # TODO: not implemented. This should fetch text features present in the data
        valid_features = categorical_featnames + continuous_featnames + language_featnames

        # TODO: Making an assumption that "feature_types_raw" above isn't used elsewhere, since feature_types_raw will
        # still have features with periods (".") in them.
        valid_features = [feat.replace(".", "/-#") for feat in valid_features]

        if len(categorical_featnames) + len(continuous_featnames) + len(language_featnames) != df.shape[1]:
            unknown_features = [feature for feature in df.columns if feature not in valid_features]

            df = df.drop(columns=unknown_features)
            self.features = list(df.columns)

        self.types_of_features=[]
        for feature in valid_features:
            if feature in categorical_featnames:
                type='CATEGORICAL'
            elif feature in continuous_featnames:
                type='SCALAR'
            elif feature in language_featnames:
                type='TEXT'
  
            self.types_of_features.append({"name": feature, "type": type})               
    

    def set_default_params(self, y_train):
        import torch

        default_params = get_default_param(self.problem_type, y_train.nunique())
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

        # TODO: Take in num_gpu's as a param. Currently this is hard-coded upon detection of cuda.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.params['device'] = device

    def get_model(self):
        self.model=TabNetClass.TabNet(self.params['n_classes'], self.params, self.cat_feat_origin_cards)
        if self.params['device'].type == "cuda":
            self.model = self.model.cuda()

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
        from .utils import TabTransformerDataset
        X = X.rename(columns = self._get_no_period_columns(X))

        if X_val is not None:
            X_val = X_val.rename(columns = self._get_no_period_columns(X_val))
        if X_unlabeled is not None:
            X_unlabeled = X_unlabeled.rename(columns = self._get_no_period_columns(X_unlabeled))

        self._get_types_of_features(X)
        
        data = TabTransformerDataset(X, col_info=self.types_of_features, **self.params)
        self.fe=fe
        if self.fe is not None:
            if X_unlabeled is None:
                unlab_data=None
            elif X_unlabeled is not None:
                unlab_data = TabTransformerDataset(X_unlabeled, col_info=self.types_of_features, **self.params)
        if self.fe is None:
            if X_unlabeled is None:
                data.fit_feat_encoders()
                self.fe = data.feature_encoders
                unlab_data=None
            elif X_unlabeled is not None:
                unlab_data = TabTransformerDataset(X_unlabeled, col_info=self.types_of_features, **self.params)
                unlab_data.fit_feat_encoders()
                self.fe = unlab_data.feature_encoders

        data.encode(self.fe)

        if X_val is not None:
            val_data = TabTransformerDataset(X_val, col_info=self.types_of_features, **self.params)
            val_data.encode(self.fe)
        else:
            val_data=None

        if unlab_data is not None:
            unlab_data.encode(self.fe)

        return data, val_data, unlab_data

    def _epoch(self, net, trainloader, valloader, y_val, optimizers, loss_criterion, pretext, state, scheduler, epoch, epochs, databar_disable, params):
        #try_import_torch()
        import torch
        from .utils import augmentation
        is_train = (optimizers is not None)
        net.train() if is_train else net.eval()
        total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(trainloader) if is_train else tqdm(valloader)

        data_bar.disable = databar_disable

        with (torch.enable_grad() if is_train else torch.no_grad()):
            for data, target in data_bar:
                data, target = pretext.get(data, target)

                if params['device'].type == "cuda":
                    data, target = data.cuda(), target.cuda()
                    pretext = pretext.cuda()

                if state in [None, 'finetune']:
                    data, target = augmentation(data, target, **params)
                    out, _    = net(data)
                elif state=='pretrain':
                    _, out    = net(data)
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

                if epochs==1:
                    train_test = 'Test'
                else:
                    train_test = 'Train'

                val_metric = None
                if valloader is not None:
                    val_metric = self.score(X=valloader, y=y_val, eval_metric=self.stopping_metric, metric_needs_y_pred=self.stopping_metric_needs_y_pred)

                if correct is not None:
                    total_correct += correct.mean().cpu().numpy()
                    data_bar.set_description('{} Epoch: [{}/{}] Train Loss: {:.4f} Validation {}: {:.2f}'.format(train_test, epoch, epochs, total_loss / total_num, self.eval_metric.name, val_metric))
                else:
                    data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f}'.format(train_test, epoch, epochs, total_loss / total_num))

            return total_loss / total_num, val_metric

        if scheduler is not None:
            scheduler.step()
        return total_loss / total_num

    def tt_fit(self, trainloader, valloader=None, y_val=None, state=None, time_limit=None):
        """
        Main training function for TabTransformer
        "state" must be one of [None, 'pretrain', 'finetune']
        None: corresponds to purely supervised learning
        pretrain: discriminative task will be a pretext task
        finetune: same as supervised learning except that the model base has
                  exponentially decaying learning rate.
        """
        try_import_torch()
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from . import pretexts

        start_time = time.time()
        pretext_tasks= pretexts.__dict__
        optimizers=[]
        lr=self.params['lr']
        weight_decay=self.params['weight_decay']
        epochs = self.params['pretrain_epochs'] if state=='pretrain' else self.params['epochs']
        freq   = self.params['pretrain_freq'] if state=='pretrain' else self.params['freq'] # TODO: What is this for?
        epochs_wo_improve = self.params['epochs_wo_improve']

        if state is None:
            optimizers = [optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)]
            pretext=pretext_tasks['SUPERVISED_pretext'](self.params)
        elif state=='pretrain':
            optimizers = [optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)]
            pretext=pretext_tasks['BERT_pretext'](self.params)
        elif state=='finetune':
            base_exp_decay=self.params['base_exp_decay']
            optimizer_fc    = optim.Adam(self.model.fc.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer_embeds = optim.Adam(self.model.embed.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer_embeds,gamma=base_exp_decay)
            optimizers.append(optimizer_fc)
            optimizers.append(optimizer_embeds)

            pretext=pretext_tasks['SUPERVISED_pretext'](self.params)

        else:
            raise NotImplementedError("state must be one of [None, 'pretrain', 'finetune']")

        if self.params['problem_type']==REGRESSION:
            loss_criterion = nn.MSELoss()
        else:
            loss_criterion = nn.CrossEntropyLoss()

        best_val_metric = -np.inf # higher = better
        best_val_epoch = 0
        val_improve_epoch = 0

        self.verbosity = self.params.get('verbosity', 5)
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

            train_loss, val_metric = self._epoch(self.model, trainloader=trainloader, valloader=valloader, y_val=y_val, optimizers=optimizers, loss_criterion=loss_criterion, \
                            pretext=pretext, state=state, scheduler=None, epoch=e, epochs=epochs, databar_disable=databar_disable, params=self.params)

            if val_metric >= best_val_metric or e == 0:
                if valloader is not None:
                    if not np.isnan(val_metric):
                        if val_metric > best_val_metric:
                            val_improve_epoch = e
                        best_val_metric = val_metric

                best_val_epoch = e
                torch.save(self.model, 'tab_trans_temp.pth')

            # If time limit has exceeded or we haven't improved in some number of epochs, stop early.
            if e - val_improve_epoch > epochs_wo_improve:
                break
            if time_limit:
                time_elapsed = time.time() - start_time
                time_left = time_limit - time_elapsed
                if time_left <= 0:
                    logger.log(20, "\tRan out of time, stopping training early.")
                    break

        if valloader is not None:
            try:
                self.model=torch.load('tab_trans_temp.pth')
                os.remove('tab_trans_temp.pth')
            except:
                pass
            logger.log(15, "Best model found in epoch %d" % best_val_epoch)

    def _fit(self, X_train, y_train, X_val=None, y_val=None, X_unlabeled=None, time_limit=None, **kwargs):
        try_import_torch()
        import torch
        self.set_default_params(y_train)

        train, val, unlab = self._tt_preprocess(X_train, X_val, X_unlabeled)

        if self.problem_type=='regression':
            train.targets = torch.FloatTensor(list(y_train))
            val.targets   = torch.FloatTensor(list(y_val))
        else:
            train.targets = torch.LongTensor(list(y_train))
            val.targets   = torch.LongTensor(list(y_val))

        trainloader, valloader, unlabloader = train.build_loader(shuffle=True), val.build_loader(), unlab.build_loader() if unlab is not None else None
   
        self.cat_feat_origin_cards = trainloader.cat_feat_origin_cards

        self.get_model()

        if X_unlabeled is not None:
            self.tt_fit(unlabloader, valloader, y_val, state='pretrain', time_limit=time_limit)
            self.tt_fit(trainloader, valloader, y_val, state='finetune', time_limit=time_limit)
        else:
            self.tt_fit(trainloader, valloader, y_val, time_limit=time_limit)

    def _predict_proba(self, X, preprocess=False):
        """
        X (torch.tensor or pd.dataframe): data for model to give prediction probabilities
        returns: np.array of k-probabilities for each of the k classes. If k=2 we drop the second probability.
        """
        try_import_torch()
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torch.autograd import Variable

        if isinstance(X, pd.DataFrame):
            if preprocess:
                X = self.preprocess(X)
            # Internal preprocessing, renaming col names, tt specific.
            X, _, _ =self._tt_preprocess(X, fe=self.fe)
            loader = X.build_loader()
        elif isinstance(X, DataLoader):
            loader = X
        elif isinstance(X, torch.Tensor):
            X = X.rename(columns = self._get_no_period_columns(X))
            loader = X.build_loader()
        else:
            raise NotImplementedError("Attempting to predict against a non-supported data type. \nNeeds to be a pandas DataFrame, torch DataLoader or torch Tensor.")

        self.model.eval()
        softmax=nn.Softmax(dim=1)

        outputs = torch.zeros([len(loader.dataset), self.num_classes])

        iter = 0
        for data, _ in loader:
            if self.params['device'].type == "cuda":
                data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
                out, _ = self.model(data)
                batch_size = len(out)
                prob = softmax(out)

            outputs[iter:(iter+batch_size)] = prob
            iter += batch_size

        if self.problem_type == BINARY:
            return outputs[:,1].cpu().numpy()

        return outputs.cpu().numpy()

    def save(self, file_prefix="", directory=None, return_filename=False, verbose=True):
        """
        file_prefix (str): Appended to beginning of file-name (does not affect directory in file-path).
        directory (str): if unspecified, use self.path as directory
        return_filename (bool): return the file-name corresponding to this save
        """
        try_import_torch()
        import torch
        if directory is not None:
            path = directory+file_prefix
        else:
            path = self.path+file_prefix

        params_filepath = path+self.params_file_name
        
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.model is not None:
            torch.save(self.model, params_filepath)

        self.model = None # Avoiding pickling the weights.
        modelobj_filepath = super().save(path=path, verbose=verbose)

        if return_filename:
            return modelobj_filepath

    @classmethod
    def load(cls, path, file_prefix="", reset_paths=False, verbose=True):
        """
        file_prefix (str): Appended to beginning of file-name.
        If you want to load files with given prefix, can also pass arg: path = directory+file_prefix
        """
        try_import_torch()
        import torch
        path = path + file_prefix
        obj: TabTransformerModel = load_pkl.load(path=path+cls.model_file_name, verbose=verbose)
        if reset_paths:
            obj.set_contexts(path)

        obj.model=torch.load(path+cls.params_file_name)

        return obj

        """
        List of features to add (Updated by Anthony Galczak 10-20-20):
        
        1) "Fix regression"
        Currently, regression has a dimensionality issue (e.g. dims [400] passed instead of [400,1]) inside the TT model.
        It also appears that the loss criterion for regression isn't actually being used.
        
        Joshs' comment on regression:
        " doesn't properly work for regression problems yet - due to discretization datapreprocessing TabTransformers
                                                                may inherantly be unsuitable for regression problems."
        
        2) Allow for saving of pretrained model for future use. This will be done in a future PR as the 
        "pretrain API change".
        
        3) Save intermediate model to "directory corresponding to specific training job". In other words, put
        'tab_trans_temp.pth' in the correct location when saving off the model.
        
        4) Investigate options for when the unlabeled schema does not match the training schema. Currently,
        we do not allow such mismatches and the schemas must match exactly. We can investigate ways to use
        less or more columns from the unlabeled data. This will likely require a design meeting.
        
        5) Enable hyperparameter tuning/searching. This will be done in a future PR.  
        """

