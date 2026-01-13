"""Example script for predicting columns of tables, demonstrating simple use-case"""

from autogluon.tabular import TabularDataset, TabularPredictor

# Training time:
train_data = TabularDataset(
    "https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv"
)  # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.head(500)  # subsample for faster demo
print(train_data.head())
label = "class"  # specifies which column do we want to predict
save_path = "ag_models/"  # where to save trained models

predictor = TabularPredictor(label=label, path=save_path).fit(train_data)
# NOTE: Default settings above are intended to ensure reasonable runtime at the cost of accuracy. To maximize predictive accuracy, do this instead:
# predictor = TabularPredictor(label=label, eval_metric=YOUR_METRIC_NAME, path=save_path).fit(train_data, presets='best_quality')
results = predictor.fit_summary(show_plot=True)

# Inference time:
test_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")  # another Pandas DataFrame
y_test = test_data[label]
test_data = test_data.drop(
    labels=[label], axis=1
)  # delete labels from test data since we wouldn't have them in practice
print(test_data.head())

predictor = TabularPredictor.load(
    save_path
)  # Unnecessary, we reload predictor just to demonstrate how to load previously-trained predictor from file
y_pred = predictor.predict(test_data)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
