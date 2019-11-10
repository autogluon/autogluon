""" Example script for PredictTableColumn task demonstrating how to use non-default arguments with HPO """

import autogluon as ag
from autogluon import PredictTableColumn as task

package_dir = 'autogluon/' # TODO: change this to absolute filepath to autogluon/ on your computer

data_dir = package_dir+'tabular/datasets/AdultIncomeData/'
train_file_path = data_dir+'train_data.csv'
test_file_path = data_dir+'test_data.csv'
savedir = data_dir+'Output/'
label_column = 'class' # name of column containing label to predict


# Training time:
train_data = task.Dataset(file_path=train_file_path) # returns Pandas object, if user already has pandas object in python, can skip this step

train_data = train_data.head(100) # subsample for faster demo
print(train_data.head())

# Call fit() with hyperparameter optimization:
nn_options = {
    'num_epochs': 10,
    'batch_size': 1024,
    'learning_rate': ag.space.Real(1e-4, 1e-2, log=True),
    'weight_decay': ag.space.Real(1e-12, 1e-1, log=True)
}

predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir, hyperparameter_tune=True, 
                     num_trials=10, time_limits=10*60, nn_options=nn_options)
# Since tuning_data = None, AutoGluon automatically determines train/validation split.

trainer = predictor.load_trainer() # use to show summary of training / HPO processes
print(trainer.model_performance)
print(trainer.model_names)
print(trainer.hpo_results)



# Inference time:
test_data = task.Dataset(file_path=test_file_path) # Pandas object
y_test = test_data[label_column]
test_data = test_data.drop(labels=[label_column],axis=1) # Delete labels from test data since we wouldn't have them in practice
print(test_data.head())

predictor = None  # We delete predictor here to demonstrate how to load previously-trained predictor from file:
predictor = task.load(savedir)

y_pred = predictor.predict(test_data)
perf = predictor.evaluate(y_true=y_test, y_pred=y_pred)

ag.done() # Turn off autogluon's remote workers. You cannot call fit() with HPO after this in the same python session.

