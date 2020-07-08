""" Example script for predicting columns of tables, demonstrating more advanced usage of fit().
    Note that all settings demonstrated here are just chosen for demonstration purposes (to minimize runtime), and do not represent wise choices to use in practice.
    To maximize predictive accuracy, we recommend you do NOT specify `hyperparameters` or `hyperparameter_tune`, and instead only specify the following fit() arguments: eval_metric=YOUR_METRIC, presets='best_quality'
"""

import autogluon as ag
from autogluon import TabularPrediction as task

# Training time:
train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv') # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.head(100) # subsample for faster demo
print(train_data.head())
label_column = 'class' # specifies which column do we want to predict
savedir = 'ag_hpo_models/' # where to save trained models

hyperparams = {'NN': {'num_epochs': 10, 'activation': 'relu', 'dropout_prob': ag.Real(0.0,0.5)},
               'GBM': {'num_boost_round': 1000, 'learning_rate': ag.Real(0.01,0.1,log=True)} }

predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir,
                     hyperparameter_tune=True, hyperparameters=hyperparams,
                     num_trials=5, time_limits=1*60, num_bagging_folds=0, stack_ensemble_levels=0) # since tuning_data = None, automatically determines train/validation split

results = predictor.fit_summary() # display detailed summary of fit() process

# Inference time:
test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv') # another Pandas DataFrame
print(test_data.head())

perf = predictor.evaluate(test_data) # shorthand way to evaluate our predictor if test-labels available

# Otherwise we make predictions and can evaluate them later:
y_test = test_data[label_column]
test_data = test_data.drop(labels=[label_column],axis=1) # Delete labels from test data since we wouldn't have them in practice
y_pred = predictor.predict(test_data)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
