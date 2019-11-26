""" Example script for  for predicting columns of tables, demonstrating how to use hyperparameter optimization """

import autogluon as ag
from autogluon import TabularPrediction as task

# Training time:
train_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification/train_data.csv') # can be local CSV file as well, returns Pandas object.
train_data = train_data.head(100) # subsample for faster demo
print(train_data.head())
label_column = 'class' # specifies which column do we want to predict
savedir = 'ag_hpo_models/' # where to save trained models

predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir, hyperparameter_tune=True, num_trials=10, time_limits=10*60)

ag.done() # Turn off autogluon's remote workers. You cannot call fit() with HPO after this within the same python session.
trainer = predictor.load_trainer() # use to show summary of training / HPO processes
print(trainer.hpo_results)
print(trainer.model_performance)


# Inference time:
test_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification/test_data.csv') # Another Pandas object
y_test = test_data[label_column]
test_data = test_data.drop(labels=[label_column],axis=1) # Delete labels from test data since we wouldn't have them in practice
print(test_data.head())

predictor = None  # We delete predictor here to demonstrate how to load previously-trained predictor from file:
predictor = task.load(savedir)

y_pred = predictor.predict(test_data)
perf = predictor.evaluate(y_true=y_test, y_pred=y_pred)

