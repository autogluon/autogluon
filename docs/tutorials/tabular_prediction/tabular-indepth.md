# Predicting Columns in a Table - In Depth

This tutorial describes how you can exert greater control over `fit()` by specifying the appropriate arguments. Let's start by loading the same census data table, and try to predict the `occupation` variable in order to demonstrate a multiclass classification problem.

```{.python .input}
import autogluon as ag
from autogluon import TabularPrediction as task

train_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification/train_data.csv') # can be local CSV file as well, returns Pandas object.
train_data = train_data.head(500) # subsample 500 data points for faster demo
print(train_data.head())

val_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification/test_data.csv')

label_column = 'occupation'
print(train_data['occupation'].describe())
```

Let's use AutoGluon to train some models, this time exerting greater control over the process via user-specified arguments. To demonstrate how you can provide your own validation dataset against which AutoGluon tunes hyperparameters, we'll use the previous test dataset as validation data this time. If you do not have any particular validation dataset of interest, we recommend omitting the `tuning_data` argument and letting AutoGluon automatically select validation data from your provided training set (as it uses smart strategies such as stratified sampling in classification problems).  For greater control, you can specify the `holdout_frac` argument to tell AutoGluon what fraction of the provided training data to hold out for validation. 

**Caution:** Since AutoGluon tunes internal knobs based on this validation data, performance estimates reported on this data may be over-optimistic. For unbiased performance estimates, you should always call `predict()` on an entirely separate dataset (that was never given to `fit()`), as we did in the previous **Quick-Start** tutorial. 

`fit()` trains neural networks and tree ensembles by default, and we can specify various hyperparameter values for each type of model. For each hyperparameter, we can either specify a single fixed value, or a search space of values to consider during the hyperparameter optimization. Hyperparameters which we do not specify are left at default settings chosen by AutoGluon, which may be fixed values or search spaces, depending on the particular hyperparameter and the setting of `hyperparameter_tune`.

```{.python .input}
hyperparameter_tune = True # whether or not to do hyperparameter optimization

nn_options = { # specifies non-default hyperparameter values for neural network models
    'num_epochs': 10, # number of training epochs (controls training time of NN models)
    'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True), # learning rate used in training (real-valued hyperparameter searched on log-scale)
    'activation': ag.space.Categorical('relu', 'softrelu', 'tanh'), # activation function used in NN (categorical hyperparameter, default = first entry)
    'layers': ag.space.Categorical([100],[1000],[200,100],[300,200,100],[400,300,200,100],[400]*3), 
      # Each choice for categorical hyperparameter 'layers' corresponds to list of sizes for each NN layer (length of list determines # of hidden layers)
    'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1), # dropout probability (real-valued hyperparameter)
}

gbm_options = { # specifies non-default hyperparameter values for lightGBM gradient boosted trees
    'num_boost_round': 100, # number of boosting rounds (controls training time of GBM models)
    'num_leaves': ag.space.Int(lower=26, upper=66, default=36), # number of leaves in trees (integer hyperparameter)
}

hyperparameters = {'NN': nn_options, 'GBM': gbm_options} # hyperparameters of each model type
# If one of these keys is missing from hyperparameters dict, then no models of that type are trained.

feature_prune = False # whether or not to perform feature selection (fit may take a while if True)
time_limits = 2*60 # train various models for under ~2 min
num_trials = 3 # try at most 3 different hyperparameter configurations for each type of model
nthreads_per_trial = 1 # use this many CPU threads per training trial (ie. evaluation of one hyperparameter configuration)
search_strategy = 'skopt' # to tune hyperparameters using SKopt Bayesian optimization routine
output_directory = 'agModels-predictOccupation' # folder where to store trained models

predictor = task.fit(train_data=train_data, tuning_data=val_data, label=label_column, output_directory=output_directory, 
                     time_limits=time_limits, num_trials=num_trials, nthreads_per_trial=nthreads_per_trial,
                     hyperparameter_tune=hyperparameter_tune, feature_prune=feature_prune,
                     hyperparameters=hyperparameters, search_strategy=search_strategy)
```

For posteriority, we again demonstrate how to use the trained models to predict on the validation data. We caution again that performance estimates from this data may be biased since it was used to tune hyperparameters.

```{.python .input}
test_data = val_data.copy()
y_test = test_data[label_column]
test_data = test_data.drop(labels=[label_column],axis=1) # delete label column to prove we are not cheating

predictor = task.load(output_directory) # unnecessary, just demonstrates how to load previously-trained predictor from file

y_pred = predictor.predict(test_data)
print("Predictions:  ", list(y_pred)[:5])
print("Actual labels:  ", list(y_test)[:5])
perf = predictor.evaluate(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
```

`predictor` can also make a prediction on just an individual example rather than a full dataset:

```{.python .input}
datapoint = test_data.iloc[[0]] # Note: .iloc[0] will not work because it returns pandas Series instead of DataFrame
print(datapoint)
y_pred = predictor.predict(datapoint)
```

In the above example, the predictive performance may be poor because we specified very little training to ensure quick runtimes.  You can call `fit()` multiple times playing with the above settings to better understand how these choices affect things.
