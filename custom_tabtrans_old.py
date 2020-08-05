""" Example script for defining and using custom models in AutoGluon Tabular """
import time

from autogluon import TabularPrediction as task
from autogluon.task.tabular_prediction.hyperparameter_configs import get_hyperparameter_config
from autogluon.utils.tabular.data.label_cleaner import LabelCleaner
from autogluon.utils.tabular.ml.models.abstract.abstract_model import AbstractModel
from autogluon.utils.tabular.ml.utils import infer_problem_type

import autogluon.utils.tabular.ml.models.tab_transformer as tabtrans
from autogluon.utils.tabular.ml.models.tab_transformer import utils, TabTransformer
from autogluon.utils.tabular.ml.models.tab_transformer.kwargs import get_kwargs

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
#########################
# Create a custom model #
#########################

class Net(nn.Module):
            def __init__(self,
                        num_class,
                        kwargs,
                        cat_feat_origin_cards):
                super(Net, self).__init__()
                self.kwargs=kwargs
                self.kwargs['cat_feat_origin_cards']=cat_feat_origin_cards
                self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                self.embed=TabTransformer.TabTransformer(**self.kwargs['tab_kwargs'], **self.kwargs)
                #self.readout = readouts.__dict__[kwargs['readout']]

                relu = nn.ReLU()
                fc  = nn.Linear(2*self.kwargs['feature_dim'] , num_class, bias=True) 
                self.fc = nn.Sequential(*[relu,fc])
     
            def forward(self, data):
                #TODO must fix
                features = self.embed(data)
                out = self.fc(features)
                return out

            def fit(self, trainloader):
            #def fit(self, X_train, y_train):
                optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)
                if self.kwargs['problem_type']=='regression':
                    loss_criterion = nn.MSE()
                else:
                    loss_criterion = nn.CrossEntropyLoss()

                """TODO: enable multigpu training
                number_devices = torch.cuda.device_count()
                number_devices = 1
                #model = nn.DataParallel(model , device_ids=list(range(number_devices))).to(device)
                """
                for e in range(self.kwargs['epochs']):
                    _ = utils.epoch(self, trainloader, optimizer, loss_criterion=loss_criterion, scheduler=None, epoch=e, epochs=self.kwargs['epochs']) #returns train_loss, train_acc@1, train_acc@5 

            #ef predict_proba(X=X, preprocess=False):
            #    if preprocess is True:
            #        pass

# In this example, we create a custom Naive Bayes model for use in AutoGluon
class TabTransformerModel(AbstractModel):

    def set_default_params(self, y_train):
        print('here')
        self.problem_type = infer_problem_type(y=y_train)  # Infer problem type (or else specify directly)
        
        if self.problem_type=='regression':
            self.num_class=1
        elif self.problem_type=='binary':
            self.num_class=2
        elif self.problem_type=='multiclass':
            self.num_class=train_dataset.num_classes
        

        #self.tabtrans_kwargs=get_kwargs()
        self.kwargs=get_kwargs(**{'problem_type': self.problem_type})

    def get_model(self, trainloader):
        self.model=Net(self.num_class, self.kwargs, trainloader.cat_feat_origin_cards)


    def preprocess(self, X, y=None, fe=None):
        X = X.select_dtypes(['category', 'object'])
        data = utils.TabTransformerDataset(X, y, **self.kwargs)

        self.fe=fe
        if self.fe is None:
            data.fit_feat_encoders()
            self.fe = data.feature_encoders

        data.encode(self.fe)
        breakpoint()
        loader=data.build_loader()
        return loader

        #self.predict(X=X, preprocess=preprocess)

        
    def predict(self, X_test):
        testloader = self.preprocess(X_test, fe=self.fe)

        preds=[]
        self.model.eval()
        with torch.no_grad():
            for data, _ in tqdm(testloader):
                data = data.to(self.model.device, non_blocking=True) 
                out  = self.model(data)
                preds.append(out)

        preds=torch.cat(preds,dim=0)
    
        preds=preds.argmax(dim=1).tolist() #convert pre-softmax predictions into hard labels 
                                  #by taking the argmax
        return preds

    def score(self, X, y):
        y_pred=self.predict(X)
        return self.eval_metric(y, y_pred)


    def _fit(self, X_train, y_train, **kwargs):
        label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=y_train)
        y_train = label_cleaner.transform(y_train)
       
        self.set_default_params(y_train)
        trainloader = self.preprocess(X_train, y_train)
        
        self.get_model(trainloader)
        self.model.fit(trainloader) #X_train, y_train)


################
# Loading Data #
################

train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
label_column = 'class'  # specifies which column do we want to predict
train_data = train_data.head(1000)  # subsample for faster demo

#############################################
# Training custom model outside of task.fit #
#############################################


# Separate features and labels
X_train = train_data.drop(columns=[label_column])
y_train = train_data[label_column]


problem_type = infer_problem_type(y=y_train)  # Infer problem type (or else specify directly)
naive_bayes_model = TabTransformerModel(path='AutogluonModels/', name='CustomTabTransformer', problem_type=problem_type)

# Construct a LabelCleaner to neatly convert labels to float/integers during model training/inference, can also use to inverse_transform back to original.
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y_train)
y_train_clean = label_cleaner.transform(y_train)

naive_bayes_model.fit(X_train=X_train, y_train=y_train_clean)  # Fit custom model


# Prepare test data
X_test = test_data.drop(columns=[label_column])
y_test = test_data[label_column]
y_test_clean = label_cleaner.transform(y_test)

y_pred = naive_bayes_model.predict(X_test)

y_pred_orig = label_cleaner.inverse_transform(y_pred)


score = naive_bayes_model.score(X_test, y_test_clean)

print(f'test score ({naive_bayes_model.eval_metric.name}) = {score}')


########################################
# Training custom model using task.fit #
########################################
"""
custom_hyperparameters = {TabTransformerModel: {}}
# custom_hyperparameters = {TabTransformerModel: [{}, {'var_smoothing': 0.00001}, {'var_smoothing': 0.000002}]}  # Train 3 TabTransformer models with different hyperparameters
predictor = task.fit(train_data=train_data, label=label_column, hyperparameters=custom_hyperparameters)  # Train a single default TabTransformerModel
predictor.leaderboard(test_data)

y_pred = predictor.predict(test_data)
print(y_pred)

time.sleep(1)  # Ensure we don't use the same train directory

###############################################################
# Training custom model alongside other models using task.fit #
###############################################################

# Now we add the custom model to be trained alongside the default models:
custom_hyperparameters.update(get_hyperparameter_config('default'))
predictor = task.fit(train_data=train_data, label=label_column, hyperparameters=custom_hyperparameters)  # Train the default models plus a single default TabTransformerModel
# predictor = task.fit(train_data=train_data, label=label_column, auto_stack=True, hyperparameters=custom_hyperparameters)  # We can even use the custom model in a multi-layer stack ensemble
predictor.leaderboard(test_data)

y_pred = predictor.predict(test_data)
print(y_pred)
"""