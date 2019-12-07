# Predicting Columns in a Table - In Depth

This tutorial describes how you can exert greater control over `fit()` by specifying the appropriate arguments. 
Let's start by loading the same census data table, and try to predict the `occupation` variable in order to demonstrate a multi-class classification problem.

```{.python .input}
import autogluon as ag
from autogluon import TabularPrediction as task

train_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/train.csv')
train_data = train_data.head(500) # subsample 500 data points for faster demo
print(train_data.head())

val_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/test.csv')

label_column = 'occupation'
print("Summary of occupation column: \n", train_data['occupation'].describe())
```

Let's use AutoGluon to train some models, this time exerting greater control over the process via user-specified arguments. To demonstrate how you can provide your own validation dataset against which AutoGluon tunes hyperparameters, we'll use the previous test dataset as validation data this time. If you do not have any particular validation dataset of interest, we recommend omitting the `tuning_data` argument and letting AutoGluon automatically select validation data from your provided training set (it uses smart strategies such as stratified sampling).  For greater control, you can specify the `holdout_frac` argument to tell AutoGluon what fraction of the provided training data to hold out for validation. 

**Caution:** Since AutoGluon tunes internal knobs based on this validation data, performance estimates reported on this data may be over-optimistic. For unbiased performance estimates, you should always call `predict()` on an entirely separate dataset (that was never given to `fit()`), as we did in the previous **Quick-Start** tutorial. We also emphasize that most options specified in this tutorial are chosen to minimize runtime for the purposes of demonstration and you should select more reasonable values in order to obtain high-quality models.
 
`fit()` trains neural networks and various types of tree ensembles by default, and we can specify various hyperparameter values for each type of model. For each hyperparameter, we can either specify a single fixed value, or a search space of values to consider during the hyperparameter optimization. Hyperparameters which we do not specify are left at default settings chosen by AutoGluon, which may be fixed values or search spaces, depending on the particular hyperparameter and the setting of `hyperparameter_tune`.

```{.python .input}
hp_tune = True  # whether or not to do hyperparameter optimization

nn_options = { # specifies non-default hyperparameter values for neural network models
    'num_epochs': 10, # number of training epochs (controls training time of NN models)
    'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True), # learning rate used in training (real-valued hyperparameter searched on log-scale)
    'activation': ag.space.Categorical('relu', 'softrelu', 'tanh'), # activation function used in NN (categorical hyperparameter, default = first entry)
    'layers': ag.space.Categorical([100],[1000],[200,100],[300,200,100]), 
      # Each choice for categorical hyperparameter 'layers' corresponds to list of sizes for each NN layer to use
    'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1), # dropout probability (real-valued hyperparameter)
}

gbm_options = { # specifies non-default hyperparameter values for lightGBM gradient boosted trees
    'num_boost_round': 100, # number of boosting rounds (controls training time of GBM models)
    'num_leaves': ag.space.Int(lower=26, upper=66, default=36), # number of leaves in trees (integer hyperparameter)
}

hyperparameters = {'NN': nn_options, 'GBM': gbm_options}  # hyperparameters of each model type
# If one of these keys is missing from hyperparameters dict, then no models of that type are trained.

time_limits = 2*60  # train various models for ~2 min
num_trials = 5  # try at most 3 different hyperparameter configurations for each type of model
search_strategy = 'skopt'  # to tune hyperparameters using SKopt Bayesian optimization routine
output_directory = 'agModels-predictOccupation'  # folder where to store trained models

predictor = task.fit(train_data=train_data, tuning_data=val_data, label=label_column,
                     output_directory=output_directory, time_limits=time_limits, num_trials=num_trials, 
                     hyperparameter_tune=hp_tune, hyperparameters=hyperparameters, 
                     search_strategy=search_strategy)
```

We again demonstrate how to use the trained models to predict on the validation data. We caution again that performance estimates from this data may be biased since it was used to tune hyperparameters.

```{.python .input}
test_data = val_data.copy()
y_test = test_data[label_column]
test_data = test_data.drop(labels=[label_column],axis=1)  # delete label column

y_pred = predictor.predict(test_data)
print("Predictions:  ", list(y_pred)[:5])
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=False)
```

`predictor` can also make a prediction on just an individual example rather than a full dataset:

```{.python .input}
datapoint = test_data.iloc[[0]]  # Note: .iloc[0] won't work because it returns pandas Series instead of DataFrame
print(datapoint)
print("Prediction:", predictor.predict(datapoint))
```

We again view a summary of what happened during fit, which this time shows details of the hyperparameter-tuning process for each type of model:

```{.python .input}
results = predictor.fit_summary()
```

In the above example, the predictive performance may be poor because we specified very little training to ensure quick runtimes.  You can call `fit()` multiple times playing with the above settings to better understand how these choices affect things (first comment out the `train_data.head` command to play with a larger dataset). To see more detailed output during `fit()`, you can also pass in the argument: `verbosity = 3`.


Performance in certain applications may be measured by different metrics than the ones AutoGluon optimizes for by default. If you know the metric that counts most in your application, you can specify it as done below:

```{.python .input}
metric = 'balanced_accuracy' # Use balanced accuracy rather than standard accuracy. You can also define your own function here, see examples in: autogluon/utils/tabular/metrics/
predictor = task.fit(train_data=train_data, label=label_column, eval_metric=metric)

performance = predictor.evaluate(val_data)
```

Beyond hyperparameter-tuning with a correctly-specified metric, two other methods to boost predictive performance are bagging and stack-ensembling.  You'll often see performance improve if you specify `num_bagging_folds` = 5-10, `stack_ensemble_levels` = 1 or 2 in the call to `fit()`, but this will increase training times.

```
predictor = task.fit(train_data=train_data, label=label_column, eval_metric=metric, num_bagging_folds=5, stack_ensemble_levels=1, hyperparameters = {'NN':{'num_epochs':5}, 'GBM':{'num_boost_round':100}})
```
