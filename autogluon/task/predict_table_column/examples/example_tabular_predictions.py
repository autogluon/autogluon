"""
Example (advanced) user script for PredictTableColumn task:

Notes to run:
source ~/virtual/TabularAutoGluon/bin/activate

"""


# Clean use-case with mostly defaults:

from autogluon import predict_table_column as task

data_dir = '/Users/jonasmue/Documents/Datasets/AdultIncomeOpenMLTask=7592/'
train_file_path = data_dir+'train_adultincomedata.csv'
test_file_path = data_dir+'test_adultincomedata.csv'
savedir = data_dir+'Output/'
label_column = 'class' # name of column containing label to predict


# Training time:
train_data = task.load_data(train_file_path) # returns Pandas object, if user already has pandas object in python, can skip this step
train_data = train_data.head(1000) # subsample for faster demo
print(train_data.head())

predictor = task.fit(train_data=train_data, label=label_column, savedir=savedir, hyperparameter_tune=False) # val=None automatically determines train/val split, otherwise we check to ensure train/val match
print(predictor.load_trainer().__dict__) # summary of training processes


# Inference time:
test_data = task.load_data(test_file_path) # Pandas object
y_test = test_data[label_column]
test_data = test_data.drop(labels=[label_column],axis=1)
print(test_data.head())

trained_predictor = task.load(savedir) # Grail object
y_pred = trained_predictor.predict(test_data)
perf = trained_predictor.evaluate(y_true=y_test, y_pred=y_pred)






































### Scratch ###

# Play with tabular NN module:

# trainer.feature_types_metadata

from autogluon import predict_table_column as task
# from f3_grail_experiments.ml.models.abstract_model import AbstractModel
from f3_grail_experiments.ml.models.nn_tab_model import NNTabularModel
from f3_grail_experiments.sandbox.models.nn.parameters import get_param_baseline as nn_get_param_baseline, get_nlp_param_baseline
from f3_grail_experiments.ml.learner.default_learner import DefaultLearner as Learner
from f3_grail_experiments.feature_generators.auto_ml_feature_generator import AutoMLFeatureGenerator

from sklearn.metrics import accuracy_score


data_dir = '/Users/jonasmue/Documents/Datasets/AdultIncomeOpenMLTask=7592/'
train_file_path = data_dir+'train_adultincomedata.csv'
test_file_path = data_dir+'test_adultincomedata.csv'
savedir = data_dir+'Output/'
label_column = 'class' # name of column containing label to predict


# Training time:
train_data = task.load_data(train_file_path) # returns Pandas object, if user already has pandas object in python, can skip this step
train_data = train_data.head(1000)
learner = Learner(path_context=savedir, label='class', submission_columns=[], feature_generator=AutoMLFeatureGenerator())
learner.fit(X=train_data)
trainer = learner.load_trainer()


mlp_model = NNTabularModel(path=savedir, name='NNTabularModel', params=nn_get_param_baseline('binary'), problem_type='binary', objective_func=accuracy_score)
mlp_model.feature_types_metadata = trainer.feature_types_metadata

mlp_model.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test)
score = model.score(X=X_test, y=y_test)

# learner = task.fit(train_data=train_data, label=label_column, savedir=savedir) # val=None automatically determines train/val split, otherwise we check to ensure train/val match












learner.predict:
- checks test data is properly formatted.
- transforms test data (gets rid of label if present with warning), unless data_transformed indicates new data is already processed
- feeds transformed test data to ensemble to get predictions
- alter categorical features with few levels to be treated as numeric after one-hot

Trainer class gets Searcher & Scheduler 
and will keep track of all models+HPconfigs and their performance.
Will need to instantiate separate searcher for each 

trainer.train() should:
- get models
- create scheduler
- create searcher for trees & train trees
- create searcher for NN & train nets

tabular.mxnet.model TODOs:
- log/box-cox transform count data
- allow preprocessors to be tunable hyperparameters

Internal Steps are:

Dataset:
1) load data into pandas via grail utilities (either using load_pd function or from parquet file).
PredictTableColumn.Dataset object will need to contain 

2) train/test split + perform pandas-based feature engineering
3) Duplicate datasets into numpy + mxnet dataloader

4) 


## TODOs:

- f3_grail_data_frame_utilities is currently installed via: 
git+ssh://git.amazon.com/pkg/F3GrailDataFrameUtilities#egg=F3GrailDataFrameUtilities
Needs to be wrapped into autogluon as another submodule.


- currently am assuming datetime features have been appropriately parsed into either numerical or categorical features.
- should eventually log-transform count features before scaling.


- How to control runtime of GRAIL?
- Need to be able to pass HPO control-args to task.fit() for both trees and NN.

- auto_ml_feature_generator.py: imputes numerical data with -1. NA is the only allowed missing value.


Integration steps:
- convert tabularNN to our mxnet model (fixed train/val split?):
a) convert fastai TabularList class
b) convert 


- integrate mxnet model with ag.scheduler/searcher
- find data with text fields to evaluate language capabilities of GRAIL
- convert pytorch language model -> gluon NLP
- integrate language model with ag.scheduler/searcher
- integrate lightGBM with ag.schedule/searcher
- add in my NN data augmentation strategies
- implement stratified sampling + better user interface
- determine how to control overall runtime of grail
- enable multiple tree/NN models in final ensemble

For now has additional dependency:
pip install category_encoders 


