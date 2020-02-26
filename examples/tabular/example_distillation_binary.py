""" Distillation example for binary classification """

import shutil, os
import numpy as np

import autogluon as ag
from autogluon import TabularPrediction as task
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS

subsample_size = 1000
hyperparameters = None # {'NN': {'num_epochs': 300}, 'GBM': {'num_boost_round': 50000}}
time_limits = 600

label_column = 'class' # specifies which column do we want to predict
savedir = 'ag_models_distillbinary/' # where to save trained models

train_filepath = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/train.csv'
test_filepath = 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/test.csv'

# Training time:
train_data = task.Dataset(file_path=train_filepath)
train_data = train_data.head(subsample_size) # subsample for faster demo
print(train_data.head())

test_data = task.Dataset(file_path=test_filepath)
test_data = test_data.head(subsample_size) # subsample for faster run

# Fit models:
# shutil.rmtree(savedir, ignore_errors=True) # Delete AutoGluon output directory to ensure previous runs' information has been removed.
predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir,
                     enable_fit_continuation=True, auto_stack=True, hyperparameters=hyperparameters,
                     verbosity=3, time_limits = time_limits, eval_metric='roc_auc')

# If you have previously-trained ensemble predictor from task.fit(),
# you can instead skip above  task.fit() and just load it:
predictor = task.load(savedir, verbosity=4) # use high-verbosity to see distillation process details.

# Distill ensemble into single model:
learner = predictor._learner

num_augmented_samples = int(3.0*len(train_data)) # distillation-training will take longer the bigger this value is, but bigger values can produce superior distilled models.
print(num_augmented_samples)

learner.augment_distill(num_augmented_samples=num_augmented_samples, time_limits=time_limits)

# Compare best compressed single model with best distilled model:
trainer = learner.load_trainer()
best_baggedbase_model = trainer.best_single_model(stack_name='core', stack_level=0)
best_compressed_model = trainer.refit_single_full(models=[best_baggedbase_model])[0]
best_distilled_model = trainer.best_single_model(stack_name='distill', stack_level=0)
print("Best compressed: %s, best distill: %s" % (best_compressed_model,best_distilled_model))
all_compressed_models = trainer.refit_single_full() # tries to refit single full version of all model types.

# Compare performance of different models on test data after distillation:
# Note: validation metrics may change during distillation so cannot compare validation metrics of distilled predictors with original predictors:
leaders_postdistill = learner.leaderboard(test_data)
# learner.save()

# Make predictions with different models:
y_pred_ensemble = learner.predict_proba(test_data)
y_pred_distill = learner.predict_proba(test_data, model=best_distilled_model)
y_pred_compressed = learner.predict_proba(test_data, model=best_compressed_model)
y_test = test_data[label_column]
ensemble_score = predictor.evaluate_predictions(y_test, y_pred_ensemble)
distill_score = predictor.evaluate_predictions(y_test, y_pred_distill)
compress_score = predictor.evaluate_predictions(y_test, y_pred_compressed)
print("ensemble_score=%s, distill_score=%s, compress_score=%s" % (ensemble_score,distill_score,compress_score))









