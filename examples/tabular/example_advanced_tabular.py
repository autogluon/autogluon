"""Example script for predicting columns of tables, demonstrating more advanced usage of fit().
Note that all settings demonstrated here are just chosen for demonstration purposes (to minimize runtime), and do not represent wise choices to use in practice.
To maximize predictive accuracy, we recommend you do NOT specify `hyperparameters` or `hyperparameter_tune_kwargs`, and instead specify `TabularPredictor(..., eval_metric=YOUR_METRIC).fit(..., presets='best_quality')`
"""

from autogluon.common import space
from autogluon.tabular import TabularDataset, TabularPredictor

# Training time:
train_data = TabularDataset(
    "https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv"
)  # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.head(100)  # subsample for faster demo
print(train_data.head())
label = "class"  # specifies which column do we want to predict
save_path = "ag_hpo_models/"  # where to save trained models

hyperparameters = {
    "NN_TORCH": {"num_epochs": 10, "activation": "relu", "dropout_prob": space.Real(0.0, 0.5)},
    "GBM": {"num_boost_round": 1000, "learning_rate": space.Real(0.01, 0.1, log=True)},
    "XGB": {"n_estimators": 1000, "learning_rate": space.Real(0.01, 0.1, log=True)},
}

predictor = TabularPredictor(label=label, path=save_path).fit(
    train_data, hyperparameters=hyperparameters, hyperparameter_tune_kwargs="auto", time_limit=60
)

results = predictor.fit_summary()  # display detailed summary of fit() process
print(results)

# Inference time:
test_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")  # another Pandas DataFrame
print(test_data.head())

perf = predictor.evaluate(test_data)  # shorthand way to evaluate our predictor if test-labels are available

# Otherwise we make predictions and can evaluate them later:
y_pred = predictor.predict_proba(test_data)
perf = predictor.evaluate_predictions(y_true=test_data[label], y_pred=y_pred, auxiliary_metrics=True)
