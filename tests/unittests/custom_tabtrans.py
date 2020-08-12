""" Example script for defining and using custom models in AutoGluon Tabular """
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
#########################
# Create a custom model #
#########################


################
# Loading Data #
################

train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame



label_column = 'class'  # specifies which column do we want to predict

unlabeled_data=train_data.drop(columns=[label_column])
train_data = train_data.head(500)  # subsample for faster demo



#############################################
# Training custom model outside of task.fit #
#############################################
"""

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
"""
########################################
# Training custom model using task.fit #
########################################

custom_hyperparameters = {"Transf": {}}
#custom_hyperparameters = {'NN': {}}
# custom_hyperparameters = {TabTransformerModel: [{}, {'var_smoothing': 0.00001}, {'var_smoothing': 0.000002}]}  # Train 3 TabTransformer models with different hyperparameters

predictor = task.fit(train_data=train_data, label=label_column, hyperparameters=custom_hyperparameters)  # Train a single default TabTransformerModel
predictor.leaderboard(test_data)
y_pred = predictor.predict(test_data)
#print(y_pred)

time.sleep(1)  # Ensure we don't use the same train directory


###############################################################
# Training custom model alongside other models using task.fit #
###############################################################

"""
custom_hyperparameters = {TabTransformerModel: {}}
# Now we add the custom model to be trained alongside the default models:
custom_hyperparameters.update(get_hyperparameter_config('default'))


#embedding = task.pretrain(unlabeled, pretrain_directory)
#embedding(X_new)
#train_data=train_data + embedding(X_new)

predictor = task.fit(train_data=train_data, label=label_column, unlabeled_data=unlabeled_data, hyperparameters=custom_hyperparameters)  # Train the default models plus a single default TabTransformerModel
# predictor = task.fit(train_data=train_data, label=label_column, auto_stack=True, hyperparameters=custom_hyperparameters)  # We can even use the custom model in a multi-layer stack ensemble
predictor.leaderboard(test_data)

y_pred = predictor.predict(test_data)
print(y_pred)
"""

