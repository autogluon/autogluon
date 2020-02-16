""" Distillation example for multi-class classification """

import shutil, os
import numpy as np

import autogluon as ag
from autogluon import TabularPrediction as task
from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS

subsample_size = 5000
hyperparameters = None # {'NN': {'num_epochs': 300}, 'GBM': {'num_boost_round': 50000}}
time_limits = 600

multi_dataset = {'url': 'https://autogluon.s3-us-west-2.amazonaws.com/datasets/CoverTypeMulticlassClassification.zip',
                  'name': 'CoverTypeMulticlassClassification',
                  'problem_type': MULTICLASS,
                  'label_column': 'Cover_Type',
                  'performance_val': 0.032} # big dataset with 7 classes, all features are numeric. Runs SLOW.

dataset = multi_dataset
directory = dataset['name'] + "/"
train_file = 'train_data.csv'
test_file = 'test_data.csv'
train_file_path = directory + train_file
test_file_path = directory + test_file

if (not os.path.exists(train_file_path)) or (not os.path.exists(test_file_path)):  # fetch files from s3:
    print("%s data not found locally, so fetching from %s" % (dataset['name'],  dataset['url']))
    os.system("wget " + dataset['url'] + " -O temp.zip && unzip -o temp.zip && rm temp.zip")

savedir = directory + 'AutogluonOutput/'
label_column = dataset['label_column']
train_data = task.Dataset(file_path=train_file_path)
test_data = task.Dataset(file_path=test_file_path)
train_data = train_data.head(subsample_size) # subsample for faster demo
test_data = test_data.head(subsample_size) # subsample for faster run
print(train_data.head())

# Fit models:
predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir, 
                     enable_fit_continuation=True, auto_stack=True, hyperparameters=hyperparameters, 
                     verbosity=3, time_limits = time_limits, eval_metric='log_loss')

# If you have previously-trained ensemble predictor from task.fit(), 
# you can instead skip above  task.fit() and just load it:
predictor = task.load(savedir, verbosity=4) # use high-verbosity to see distillation process details.

# Distill ensemble into single model:
learner = predictor._learner

num_augmented_samples = max(100000, 2*len(train_data)) # distillation-training will take longer the bigger this value is, but bigger values can produce superior distilled models.
learner.augment_distill(num_augmented_samples=num_augmented_samples, time_limits=time_limits)

# Compare best compressed single model with best distilled model:
trainer = learner.load_trainer()
best_baggedbase_model = trainer.best_single_model(stack_name='core', stack_level=0)
best_compressed_model = trainer.compress(models=[best_baggedbase_model])[0]
best_distilled_model = trainer.best_single_model(stack_name='distill', stack_level=0)
print("Best compressed: %s, best distill: %s" % (best_compressed_model,best_distilled_model))

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
