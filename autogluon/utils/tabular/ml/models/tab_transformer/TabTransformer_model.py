""" TabTransformer model """
import time

from ..abstract.abstract_model import AbstractModel
from ....ml.utils import infer_problem_type
from ....utils.loaders import load_pkl

from autogluon.utils.tabular.ml.models.tab_transformer import utils
from autogluon.utils.tabular.ml.models.tab_transformer.TabTransformer import TabTransformer, TabTransformer_fix_attention
from autogluon.utils.tabular.ml.models.tab_transformer import pretexts
from autogluon.utils.tabular.ml.models.tab_transformer.hyperparameters.kwargs import get_kwargs

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd
import os

class TabNet(nn.Module):
            def __init__(self,
                        num_class,
                        kwargs,
                        cat_feat_origin_cards):
                super(TabNet, self).__init__()
                self.kwargs=kwargs
                self.kwargs['cat_feat_origin_cards']=cat_feat_origin_cards
        
              
                if self.kwargs['fix_attention'] is True:
                    self.embed=TabTransformer_fix_attention(**self.kwargs['tab_kwargs'], **self.kwargs)
                else:
                    self.embed=TabTransformer(**self.kwargs['tab_kwargs'], **self.kwargs)

                relu, lin = nn.ReLU(), nn.Linear(2*self.kwargs['feature_dim'] , num_class, bias=True) 
                self.fc = nn.Sequential(*[relu,lin])
     
            def forward(self, data):
                features = self.embed(data)
                out = features.mean(dim=1)
                out = self.fc(out)
                return out, features

            def fit(self, trainloader, valloader=None, state=None):
                """
                Main training function for TabTransformer
                "state" must be one of [None, 'pretrain', 'finetune']
                None: corresponds to purely supervised learning
                pretrain: discirminative task will be a pretext task
                finetune: same as superised learning except that the model base has 
                          exponentially decaying learning rate.
                """

                pretext_tasks=pretexts.__dict__
                optimizers=[]
                if state==None:
                    optimizers = [optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)]
                    pretext=pretext_tasks['SUPERVISED_pretext'](self.kwargs)

                elif state=='pretrain':
                    optimizers = [optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)]
                    pretext=pretext_tasks['BERT_pretext'](self.kwargs)
                elif state=='finetune':
                    optimizer_fc    = optim.Adam(self.fc.parameters(), lr=1e-3, weight_decay=1e-6) 
                    optimizer_embeds = optim.Adam(self.embed.parameters(), lr=1e-3, weight_decay=1e-6) 
                    scheduler = optim.lr_scheduler.ExponentialLR(optimizer_embeds,gamma=0.95)
                    optimizers.append(optimizer_fc)
                    optimizers.append(optimizer_embeds)

                    pretext=pretext_tasks['SUPERVISED_pretext'](self.kwargs)

                else:
                    raise NotImplementedError("state must be one of [None, 'pretrain', 'finetune']")

                if self.kwargs['problem_type']=='regression':
                    loss_criterion = nn.MSELoss()
                else:
                    loss_criterion = nn.CrossEntropyLoss()

                old_val_accuracy=0.0
         
                epochs = self.kwargs['pretrain_epochs'] if state=='pretrain' else self.kwargs['epochs']
                freq   = self.kwargs['pretrain_freq'] if state=='pretrain' else self.kwargs['freq']
   
                for e in range(1,epochs+1):
                    _ = utils.epoch(self, trainloader, optimizers, loss_criterion=loss_criterion, \
                                        pretext=pretext, state=state, scheduler=None, epoch=e, epochs=epochs) #returns train_loss, train_acc@1, train_acc@5 
                    
                    if valloader is not None:
                        if e % freq == 0:
                            _, val_accuracy = utils.epoch(self, valloader, optimizers=None, \
                                loss_criterion=loss_criterion, pretext=pretext, state=state, scheduler=None, epoch=1, epochs=1)                       
                            if val_accuracy>old_val_accuracy:
                                torch.save(self,'tab_trans_temp.pth')
                if valloader is not None:
                    try:
                        self=torch.load('tab_trans_temp.pth')
                        os.remove('tab_trans_temp.pth')
                    except:
                        pass        

            
class TabTransformerModel(AbstractModel):
    params_file_name="tab_trans_params.pth"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.types_of_features=None


    def _get_types_of_features(self, df):
        """ Returns dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = ordered list of feature-names falling into each category.
            Each value is a list of feature-names corresponding to columns in original dataframe.
            TODO: ensure features with zero variance have already been removed before this function is called.
        """
        if self.types_of_features is not None:
            Warning("Attempting to _get_types_of_features for TabTransformerModel, but previously already did this.")

        feature_types = self.feature_types_metadata.feature_types_raw

        categorical_featnames = feature_types['category'] + feature_types['object'] + feature_types['bool']
        continuous_featnames = feature_types['float'] + feature_types['int']  # + self.__get_feature_type_if_present('datetime')
        language_featnames = [] # TODO: not implemented. This should fetch text features present in the data
        valid_features = categorical_featnames + continuous_featnames + language_featnames
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
        if self.problem_type is None:
            self.problem_type = infer_problem_type(y=y_train)  # Infer problem type (or else specify directly)
        if self.problem_type=='regression':
            self.num_class=1
        elif self.problem_type=='binary':
            self.num_class=2
        elif self.problem_type=='multiclass':
            self.num_class=train_dataset.num_classes

        device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")

        #gets default kwargs for TabTransformer model.
        self.kwargs=get_kwargs(**{'problem_type': self.problem_type, 'n_classes': self.num_class, 'device': device})

    def get_model(self):
        self.model=TabNet(self.num_class, self.kwargs, self.cat_feat_origin_cards)


    def preprocess(self, X, X_val=None, X_unlabeled=None, fe=None):
        #X = X.select_dtypes(['category', 'object'])
        self._get_types_of_features(X)
        
        data = utils.TabTransformerDataset(X, col_info=self.types_of_features, **self.kwargs)
        self.fe=fe
        if self.fe is not None:
            if X_unlabeled is None:
                unlab_data=None
            elif X_unlabeled is not None:
                unlab_data = utils.TabTransformerDataset(X_unlabeled, col_info=self.types_of_features, **self.kwargs)
        if self.fe is None:
            if X_unlabeled is None:
                data.fit_feat_encoders()
                self.fe = data.feature_encoders
                unlab_data=None
            elif X_unlabeled is not None:
                unlab_data = utils.TabTransformerDataset(X_unlabeled, col_info=self.types_of_features, **self.kwargs)
                unlab_data.fit_feat_encoders()
                self.fe = unlab_data.feature_encoders


        data.encode(self.fe)

        if X_val is not None:
            val_data = utils.TabTransformerDataset(X_val, col_info=self.types_of_features, **self.kwargs)
            val_data.encode(self.fe)
        else:
            val_data=None

        if unlab_data is not None:
            unlab_data.encode(self.fe)

        return data, val_data, unlab_data



    def _fit(self, X_train, y_train, X_val=None, y_val=None, X_unlabeled=None, **kwargs):
        self.set_default_params(y_train)

        train, val, unlab = self.preprocess(X_train, X_val, X_unlabeled)
   
        if self.problem_type=='regression':
            train.targets = torch.FloatTensor(list(y_train))
            val.targets   = torch.FloatTensor(list(y_val))
        else:
            train.targets = torch.LongTensor(list(y_train))
            val.targets   = torch.LongTensor(list(y_val))

        trainloader, valloader, unlabloader = train.build_loader(), val.build_loader(), unlab.build_loader() if unlab is not None else None
   
        self.cat_feat_origin_cards=trainloader.cat_feat_origin_cards
       
        self.get_model()
        if X_unlabeled is not None:
            self.model.fit(unlabloader, valloader, state='pretrain')
            self.model.fit(trainloader, valloader, state='finetune')
        else:
            self.model.fit(trainloader, valloader) #X_train, y_train)
         

    def predict_proba(self, X, preprocess=False):
        """
        X (torch.tensor or pd.dataframe): data for model to give prediction probabilities
        returns: np.array of k-probabilities for each of the k classes. If k=2 we drop the second probability.
        """
        if preprocess or isinstance(X,pd.DataFrame):
            X, _, _ =self.preprocess(X, fe=self.fe)
        else:
            X=X
     
        self.model.eval()

        softmax=nn.Softmax(dim=1)
        with torch.no_grad():
            out, _ = self.model(X.cat_data)
            prob=softmax(out)
        prob=prob.detach().numpy()

        if prob.shape[1]==2:
            prob=prob[:,1]
        
        return prob

    def save(self, file_prefix="", directory=None, return_filename=False, verbose=True):
        """
        file_prefix (str): Appended to beginning of file-name (does not affect directory in file-path).
        directory (str): if unspecified, use self.path as directory
        return_filename (bool): return the file-name corresponding to this save
        """
        if directory is not None:
            path = directory+file_prefix
        else:
            path = self.path+file_prefix

        params_filepath = path+self.params_file_name
        
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.model is not None:
            torch.save(self.model, params_filepath)

        modelobj_filepath = super().save(file_prefix=file_prefix, directory=directory, return_filename=True, verbose=verbose)
        
        if return_filename:
            return modelobj_filepath



    @classmethod
    def load(cls, path, file_prefix="", reset_paths=False, verbose=True):
        """
        file_prefix (str): Appended to beginning of file-name.
        If you want to load files with given prefix, can also pass arg: path = directory+file_prefix
        """
        path = path + file_prefix
        obj: TabTransformerModel = load_pkl.load(path=path+cls.model_file_name, verbose=verbose)
        if reset_paths:
            obj.set_contexts(path)

        obj.model=torch.load(path+cls.params_file_name)

        return obj

        """
        list of features to add / lmitations: 
            doesn't properly work for regression problems yet
            
            training of TabTransformer currently saves intermediate model simply to 'tab_trans_temp.pth'
            Should move this to be saved in the directory corresponding to the specific training job.

            Not connected to HPO functionaity yet.
        """





