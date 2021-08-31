""" Example script for quantile regression with tabular data, demonstrating simple use-case """

from autogluon.tabular import TabularDataset, TabularPredictor

# Training time:
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.head(500)  # subsample for faster demo
print(train_data.head())

label = 'age'  # specifies which column do we want to predict
save_path = 'ag_models/'  # where to save trained models
quantiles_topredict = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # which quantiles of numeric label-variable we want to predict

predictor = TabularPredictor(label=label, path=save_path, problem_type='quantile', quantile_levels=quantiles_topredict)
predictor.fit(train_data, num_bag_folds=5)  # time_limit is optional, you should increase it for real applications

# Inference time:
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
predictor = TabularPredictor.load(save_path)  # Unnecessary, we reload predictor just to demonstrate how to load previously-trained predictor from file
y_pred = predictor.predict(test_data)
print(y_pred)  # each column contains estimates for one target quantile-level

ldr = predictor.leaderboard(test_data)  # evaluate performance of every trained model
print(f"Quantile-regression evaluated using metric = {predictor.eval_metric}")
