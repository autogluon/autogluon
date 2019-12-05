""" Example script for predicting columns of tables, demonstrating simple use-case """

from autogluon import TabularPrediction as task

# Training time:
train_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification/train_data.csv') # can be local CSV file as well, returns Pandas object.
train_data = train_data.head(500) # subsample for faster demo
print(train_data.head())
label_column = 'class' # specifies which column do we want to predict
savedir = 'ag_models/' # where to save trained models

predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir) # Since tuning_data = None, AutoGluon automatically determines train/validation split.
results = predictor.fit_summary() # display summary of models trained during fit()

# Inference time:
test_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification/test_data.csv') # Another Pandas object
y_test = test_data[label_column]
test_data = test_data.drop(labels=[label_column],axis=1) # Delete labels from test data since we wouldn't have them in practice
print(test_data.head())

predictor = task.load(savedir) # Unnecessary, we reload predictor here just to demonstrate how to load previously-trained predictor from file.
y_pred = predictor.predict(test_data)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
