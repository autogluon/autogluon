# Predicting Columns in a Table - In Depth
:label:`sec_tabularadvanced`

**Tip**: If you are new to AutoGluon, review :ref:`sec_tabularquick` to learn the basics of the AutoGluon API.

This tutorial describes how you can exert greater control when using AutoGluon's `fit()` by specifying the appropriate arguments. Using the same census data table as :ref:`sec_tabularquick`, we will try to predict the `occupation` of an individual - a multi-class classification problem.

Start by importing AutoGluon, specifying TabularPrediction as the task, and loading the data.

```{.python .input}
import autogluon as ag
from autogluon import TabularPrediction as task

train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
train_data = train_data.head(500) # subsample 500 data points for faster demo (comment this out to run on full dataset instead)
print(train_data.head())

val_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

label_column = 'occupation'
print("Summary of occupation column: \n", train_data['occupation'].describe())
```

 To demonstrate how you can provide your own validation dataset against which AutoGluon tunes hyperparameters, we'll use the test dataset from the previous tutorial as validation data.

 If you don't have a strong reason to provide your own validation dataset, we recommend you omit the `tuning_data` argument. This lets AutoGluon automatically select validation data from your provided training set (it uses smart strategies such as stratified sampling).  For greater control, you can specify the `holdout_frac` argument to tell AutoGluon what fraction of the provided training data to hold out for validation.

**Caution:** Since AutoGluon tunes internal knobs based on this validation data, performance estimates reported on this data may be over-optimistic. For unbiased performance estimates, you should always call `predict()` on a separate dataset (that was never passed to `fit()`), as we did in the previous **Quick-Start** tutorial. We also emphasize that most options specified in this tutorial are chosen to minimize runtime for the purposes of demonstration and you should select more reasonable values in order to obtain high-quality models.

`fit()` trains neural networks and various types of tree ensembles by default. You can specify various hyperparameter values for each type of model. For each hyperparameter, you can either specify a single fixed value, or a search space of values to consider during the hyperparameter optimization. Hyperparameters which you do not specify are left at default settings chosen automatically by AutoGluon, which may be fixed values or search spaces.

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

We again demonstrate how to use the trained models to predict on the validation data (We caution again that performance estimates here are biased because the same data was used to tune hyperparameters).

```{.python .input}
test_data = val_data.copy()
y_test = test_data[label_column]
test_data = test_data.drop(labels=[label_column],axis=1)  # delete label column

y_pred = predictor.predict(test_data)
print("Predictions:  ", list(y_pred)[:5])
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=False)
```

Use the following to view a summary of what happened during fit. This command will shows details of the hyperparameter-tuning process for each type of model:

```{.python .input}
results = predictor.fit_summary()
```

In the above example, the predictive performance may be poor because we specified very little training to ensure quick runtimes.  You can call `fit()` multiple times while modifying the above settings to better understand how these choices affect performance outcomes. For example: you can comment out the `train_data.head` command to train using a larger dataset, increase the `time_limits`, and increase the `num_epochs` and `num_boost_round` hyperparameters.  To see more detailed output during the execution of `fit()`, you can also pass in the argument: `verbosity = 3`.


## Specifying performance metrics

Performance in certain applications may be measured by different metrics than the ones AutoGluon optimizes for by default. If you know the metric that counts most in your application, you can specify it as done below to utilize the balanced accuracy metric instead of standard accuracy (the default):

```{.python .input}
metric = 'balanced_accuracy'
predictor = task.fit(train_data=train_data, label=label_column, eval_metric=metric,
                     output_directory=output_directory, time_limits=60)

performance = predictor.evaluate(val_data)
```

Some other non-default metrics you might use include things like: `f1` (for binary classification), `roc_auc` (for binary classification), `log_loss` (for classification), `mean_absolute_error` (for regression), `median_absolute_error` (for regression).  You can also define your own custom metric function, see examples in the folder: `autogluon/utils/tabular/metrics/`


## Model ensembling with stacking/bagging

Beyond hyperparameter-tuning with a correctly-specified evaluation metric, two other methods to boost predictive performance are bagging and stack-ensembling.  You'll often see performance improve if you specify `num_bagging_folds` = 5-10, `stack_ensemble_levels` = 1-3 in the call to `fit()`, but this will increase training times.

```{.python .input}
predictor = task.fit(train_data=train_data, label=label_column, eval_metric=metric,
                     num_bagging_folds=5, stack_ensemble_levels=1,
                     hyperparameters = {'NN':{'num_epochs':5}, 'GBM':{'num_boost_round':100}})
```

You should not provide `tuning_data` when stacking/bagging, and instead provide all your available data as `train_data` (which AutoGluon will split in more intellgent ways). Rather than manually searching for good bagging/stacking values yourself, AutoGluon will automatically select good values for you if you specify `auto_stack` instead:

```{.python .input}
predictor = task.fit(train_data=train_data, label=label_column, eval_metric=metric, auto_stack=True,
                     hyperparameters = {'NN':{'num_epochs':5}, 'GBM':{'num_boost_round':100}}, time_limits = 60) # last 2 arguments are just for quick demo, should be omitted
```


## Getting predictions (inference-time options)

Even if you've started a new Python session since last calling `fit()`, you can still load a previously trained predictor from disk:

```{.python .input}
predictor = task.load(output_directory)
```

Here, `output_directory` is the same folder previously passed to `fit()`, in which all the trained models have been saved.
You can train easily models on one machine and deploy them on another. Simply copy the `output_directory` folder to the new machine and specify its new path in `task.load()`.

`predictor` can make a prediction on an individual example rather than a full dataset:

```{.python .input}
datapoint = test_data.iloc[[0]]  # Note: .iloc[0] won't work because it returns pandas Series instead of DataFrame
print(datapoint)
print(predictor.predict(datapoint))
```

To output predicted class probabilities instead of predicted classes, you can use:

```{.python .input}
class_probs = predictor.predict_proba(datapoint)
print(class_probs)
```

By default, `predict()` and `predict_proba()` will utilize the model that AutoGluon thinks is most accurate, which is usually an ensemble of many individual models.
We can instead specify a particular model to use for predictions (e.g. to reduce inference latency).  Before deciding which model to use, let's evaluate all of the models AutoGluon has previously trained using our validation dataset:

```{.python .input}
results = predictor.leaderboard(val_data)
```

Here's how to specify a particular model to use for prediction instead of AutoGluon's default model-choice:

```{.python .input}
i = 0  # index of model to use
model_to_use = predictor.model_names[i]
model_pred = predictor.predict(datapoint, model=model_to_use)
print("Prediction from %s model: %s" % (model_to_use, model_pred))
```

The `predictor` also remembers what metric predictions should be evaluated with, which can be done with ground truth labels as follows:

```
y_pred = predictor.predict(test_data)
predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
```

However, you must be careful here as certain metrics require predicted probabilities rather than classes.
Since the label columns remains in the `val_data` DataFrame, we can instead use the shorthand:

```
predictor.evaluate(val_data)
```

which will correctly select between `predict()` or `predict_proba()` depending on the evaluation metric.


## Maximizing predictive performance

To get the best predictive accuracy with AutoGluon, you should generally use it like this:

```{.python .input}
long_time = 60 # for quick demonstration only, you should set this to longest time you are willing to wait
predictor = task.fit(train_data=train_data, label=label_column, eval_metric=metric, auto_stack=True, time_limits=long_time)
```

This command implements the following strategy to maximize accuracy:

- Specify the `auto_stack` argument, which allows AutoGluon to automatically construct model ensembles based on multi-layer stack ensembling with repeated bagging, and will greatly improve the resulting predictions if granted sufficient training time.

- Provide the `eval_metric` if you know what metric will be used to evaluate predictions in your application (e.g. `roc_auc`, `log_loss`, `mean_absolute_error`, etc.)

- Include all your data in `train_data` and do not provide `tuning_data` (AutoGluon will split the data more intelligently to fit its needs).

- Do not specify the `hyperparameter_tune` argument (counterintuitively, hyperparameter tuning is not the best way to spend a limited training time budgets, as model ensembling is often superior). We recommend you only use `hyperparameter_tune` if your goal is to deploy a single model rather than an ensemble.

- Do not specify `hyperparameters` argument (allow AutoGluon to adaptively select which models/hyperparameters to use).

- Set `time_limits` to the longest amount of time (in seconds) that you are willing to wait. AutoGluon's predictive performance improves the longer `fit()` is allowed to run.







