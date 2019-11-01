""" Example script for PredictTableColumn task """


# Clean use-case with mostly defaults:

from autogluon import PredictTableColumn as task 

package_dir = 'autogluon/' # TODO: change this to absolute filepath to autogluon/ on your computer

data_dir = package_dir+'tabular/datasets/AdultIncomeData/'
train_file_path = data_dir+'train_data.csv'
test_file_path = data_dir+'test_data.csv'
savedir = data_dir+'Output/'
label_column = 'class' # name of column containing label to predict


# Training time:
train_data = task.Dataset(file_path=train_file_path) # returns Pandas object, if user already has pandas object in python, can skip this step

train_data = train_data.head(1000) # subsample for faster demo
print(train_data.head())

predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir, hyperparameter_tune=False) 
# Since tuning_data = None, AutoGluon automatically determines train/validation split.


print(predictor.load_trainer().__dict__) # summary of training processes


# Inference time:
test_data = task.Dataset(file_path=test_file_path) # Pandas object
y_test = test_data[label_column]
test_data = test_data.drop(labels=[label_column],axis=1) # Delete labels from test data since we wouldn't have them in practice
print(test_data.head())

predictor = None  # We delete predictor here to demonstrate how to load previously-trained predictor from file:
predictor = task.load(savedir)

y_pred = predictor.predict(test_data)
perf = predictor.evaluate(y_true=y_test, y_pred=y_pred)

