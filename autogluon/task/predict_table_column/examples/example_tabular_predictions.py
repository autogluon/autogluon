""" Example script for PredictTableColumn task, demonstrating simple use-case with mostly default settings """

# Download example dataset:

import os

data_dir = 'AdultIncomeBinaryClassification'
data_url = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification.zip'
if (not os.path.exists(data_dir)) or (not os.path.exists(data_dir)):
    os.system("wget " + data_url + " -O temp.zip && unzip -o temp.zip && rm temp.zip")

train_file_path = data_dir+'/train_data.csv'
test_file_path = data_dir+'/test_data.csv'
savedir = data_dir+'/Output/'
label_column = 'class' # specify name of column containing labels to predict


# Training time:

from autogluon import PredictTableColumn as task

train_data = task.Dataset(file_path=train_file_path) # returns Pandas object, if user already has pandas object in python, can skip this step
train_data = train_data.head(1000) # subsample for faster demo
print(train_data.head())

predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir, hyperparameter_tune=False) 
# Since tuning_data = None, AutoGluon automatically determines train/validation split.

print(predictor.load_trainer().model_performance) # summarize validation performance of various models tried during fit()


# Inference time:
test_data = task.Dataset(file_path=test_file_path) # Pandas object
y_test = test_data[label_column]
test_data = test_data.drop(labels=[label_column],axis=1) # Delete labels from test data since we wouldn't have them in practice
print(test_data.head())

predictor = None  # We delete predictor here to demonstrate how to load previously-trained predictor from file:
predictor = task.load(savedir)

y_pred = predictor.predict(test_data)
perf = predictor.evaluate(y_true=y_test, y_pred=y_pred)

