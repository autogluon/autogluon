""" Basic test script for TabTransformer"""
import time

from autogluon import TabularPrediction as task
from autogluon.task.tabular_prediction.hyperparameter_configs import get_hyperparameter_config
from autogluon.utils.tabular.data.label_cleaner import LabelCleaner

from autogluon.utils.tabular.ml.models.tab_transformer.TabTransformer_model import TabTransformerModel

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd
import os


################
# Loading Data #
################

train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame


label_column='class'  # specifies which column do we want to predict

unlabeled_data=train_data.drop(columns=[label_column])
train_data = train_data.head(500)  # subsample for faster demo


####################################################
# Training only TabTransformer on supervised problem 
####################################################

custom_hyperparameters = {"Transf": {}}

predictor = task.fit(train_data=train_data, label=label_column, \
					 hyperparameters=custom_hyperparameters)  # Train a single default TabTransformerModel
predictor.leaderboard(test_data)
y_pred = predictor.predict(test_data)
print(y_pred)

time.sleep(1)  # Ensure we don't use the same train directory


######################################################################
# Training TabTransformer alongside other models on supervised problem 
######################################################################

predictor = task.fit(train_data=train_data, label=label_column)  
predictor.leaderboard(test_data)

y_pred = predictor.predict(test_data)
print(y_pred)


#########################################################
# Training only TabTransformer on semi-supervised problem 
#########################################################

custom_hyperparameters = {"Transf": {}}

predictor = task.fit(train_data=train_data, label=label_column, \
					 unlabeled_data=unlabeled_data, hyperparameters=custom_hyperparameters)  # Train a single default TabTransformerModel
predictor.leaderboard(test_data)
y_pred = predictor.predict(test_data)
print(y_pred)

time.sleep(1)  # Ensure we don't use the same train directory


###########################################################################
# Training TabTransformer alongside other models on semi-supervised problem 
###########################################################################

predictor = task.fit(train_data=train_data, label=label_column, unlabeled_data=unlabeled_data)  
predictor.leaderboard(test_data)

y_pred = predictor.predict(test_data)
print(y_pred)


