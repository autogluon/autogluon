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

