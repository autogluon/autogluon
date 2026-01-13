"""Example script for quantile regression with tabular data, demonstrating simple use-case"""

import numpy as np

from autogluon.tabular import TabularDataset, TabularPredictor

# Training time:
train_data = TabularDataset(
    "https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv"
)  # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.head(1000)  # subsample for faster demo
print(train_data.head())

label = "age"  #  which column we want to predict
save_path = "ag_models/"  # where to save trained models
quantile_levels = [0.1, 0.5, 0.9]  # which quantiles of numeric label-variable we want to predict

predictor = TabularPredictor(label=label, path=save_path, problem_type="quantile", quantile_levels=quantile_levels)
predictor.fit(
    train_data, calibrate=True, num_bag_folds=5
)  # here we fit with 5-fold bagging and calibrate quantile estimates via conformal method

# Inference time:
test_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")  # another Pandas DataFrame
test_data = test_data.head(1000)  # subsample for faster demo
predictor = TabularPredictor.load(
    save_path
)  # unnecessary here, we just to demonstrate how to load previously-trained predictor from file
y_pred = predictor.predict(test_data)
print(y_pred)  # each column contains estimates of a particular quantile-level of the label variable

# Check coverage of prediction intervals (ie. how often they contain the observed Y value):
num_quantiles = len(quantile_levels)
y_pred = y_pred.to_numpy()
y_target = test_data[label].to_numpy()
for i in range(num_quantiles // 2):
    low_idx = i
    high_idx = num_quantiles - i - 1
    low_quantile = quantile_levels[low_idx]  # which quantile to use for lower end of prediction interval
    high_quantile = quantile_levels[high_idx]  # which quantile to use for upper end of prediction interval
    pred_coverage = np.mean((y_pred[:, low_idx] <= y_target) & (y_pred[:, high_idx] >= y_target))
    target_coverage = high_quantile - low_quantile
    print(
        "Desired coverage = {:.2f} => Actual coverage of predicted [quantile {}, quantile {}] intervals over test data = {:.2f}".format(
            target_coverage, low_quantile, high_quantile, pred_coverage
        )
    )

# Evaluate performance of every trained model:
print(f"Quantile-regression evaluated using metric = {predictor.eval_metric}")
ldr = predictor.leaderboard(test_data)
