""" Example script for predicting columns of tables, demonstrating simple use-case """

from autogluon import TabularPrediction as task

# Training time:
train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.head(500) # subsample for faster demo
print(train_data.head())
label_column = 'class' # specifies which column do we want to predict
savedir = 'ag_models/' # where to save trained models

predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir)
# NOTE: Default settings above are intended to ensure reasonable runtime at the cost of accuracy. To maximize predictive accuracy, do this instead:  predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir, presets='best_quality', eval_metric=YOUR_METRIC_NAME)
results = predictor.fit_summary()

# Inference time:
test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv') # another Pandas DataFrame
y_test = test_data[label_column]
test_data = test_data.drop(labels=[label_column],axis=1) # delete labels from test data since we wouldn't have them in practice
print(test_data.head())

predictor = task.load(savedir) # Unnecessary, we reload predictor just to demonstrate how to load previously-trained predictor from file
y_pred = predictor.predict(test_data)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
