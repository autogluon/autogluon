""" Example script for PredictTableColumn task demonstrating how to use non-default arguments 
    in the hyperparameter optimization.
"""

# Download example dataset:

import os

data_dir = 'AdultIncomeBinaryClassification'
data_url = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification.zip'
if (not os.path.exists(data_dir)) or (not os.path.exists(data_dir)):
    os.system("wget " + data_url + " -O temp.zip && unzip -o temp.zip && rm temp.zip")

train_file_path = data_dir+'/train_data.csv'
test_file_path = data_dir+'/test_data.csv'
savedir = data_dir+'/Output/'
label_column = 'class' # name of column containing label to predict


# Training time:

import autogluon as ag
from autogluon import PredictTableColumn as task

train_data = task.Dataset(file_path=train_file_path) # returns Pandas object, if user already has pandas object in python, can skip this step
train_data = train_data.head(100) # subsample for faster demo
print(train_data.head())

# Call fit() with hyperparameter optimization that uses the search spaces specified below.
# Note that if hyperparameter_tune=True: AutoGluon uses default search spaces for any hyperparameters you did not specify below.
nn_options = {
    'num_epochs': 10,
    'batch_size': 1024,
    'learning_rate': ag.space.Real(1e-4, 1e-2, log=True),
    'weight_decay': ag.space.Real(1e-12, 1e-1, log=True)
}

gbm_options = {
    'num_boost_round': 1000,
}

predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir, hyperparameter_tune=True, 
                     num_trials=10, time_limits=10*60, hyperparameters={'GBM': gbm_options, 'NN':nn_options})
# Since tuning_data = None, AutoGluon automatically determines train/validation split.

ag.done() # Turn off autogluon's remote workers. You cannot call fit() with HPO after this within the same python session.
trainer = predictor.load_trainer() # use to show summary of training / HPO processes
print(trainer.hpo_results)
print(trainer.model_performance)


# Inference time:
test_data = task.Dataset(file_path=test_file_path) # Pandas object
y_test = test_data[label_column]
test_data = test_data.drop(labels=[label_column],axis=1) # Delete labels from test data since we wouldn't have them in practice
print(test_data.head())

predictor = None  # We delete predictor here to demonstrate how to load previously-trained predictor from file:
predictor = task.load(savedir)

y_pred = predictor.predict(test_data)
perf = predictor.evaluate(y_true=y_test, y_pred=y_pred)

