# Predicting Columns in a Table - In Depth
:label:`sec_tabularadvanced`

**Tip**: If you are new to AutoGluon, review :ref:`sec_tabularquick` to learn the basics of the AutoGluon API.

This tutorial describes how you can exert greater control when using AutoGluon's `fit()` by specifying the appropriate arguments. Using the same census data table as :ref:`sec_tabularquick`, we will try to predict the `occupation` of an individual - a multi-class classification problem.

Start by importing AutoGluon, specifying TabularPrediction as the task, and loading the data.

```{.python .input}
import autogluon as ag
from autogluon import TabularPrediction as task

import pandas as pd
import numpy as np

train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
subsample_size = 500 # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data.head(subsample_size)
print(train_data.head())

label_column = 'occupation'
print("Summary of occupation column: \n", train_data['occupation'].describe())

val_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
test_data = val_data[5000:].copy() # this should be separate data in your applications
y_test = test_data[label_column]
test_data_nolabel = test_data.drop(labels=[label_column],axis=1)  # delete label column
val_data = val_data[:5000]

metric = 'accuracy' # we specify eval-metric just for demo (unnecessary as it's the default)
```

 To demonstrate how you can provide your own validation dataset against which AutoGluon tunes hyperparameters, we'll use some of the test data from the previous tutorial as validation data.

 If you don't have a strong reason to provide your own validation dataset, we recommend you omit the `tuning_data` argument. This lets AutoGluon automatically select validation data from your provided training set (it uses smart strategies such as stratified sampling).  For greater control, you can specify the `holdout_frac` argument to tell AutoGluon what fraction of the provided training data to hold out for validation. One reason you may specify validation data is when future test data will stem from a different distribution than training data (and your specified validation data is more representative of the future data that will likely be encountered).

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

hyperparameters = { # hyperparameters of each model type
                   'GBM': gbm_options,
                   'NN': nn_options, # NOTE: comment this line out if you get errors on Mac OSX
                  } # When these keys are missing from hyperparameters dict, no models of that type are trained

time_limits = 2*60  # train various models for ~2 min
num_trials = 5  # try at most 3 different hyperparameter configurations for each type of model
search_strategy = 'skopt'  # to tune hyperparameters using SKopt Bayesian optimization routine

predictor = task.fit(train_data=train_data, tuning_data=val_data, label=label_column,
                     time_limits=time_limits, eval_metric=metric, num_trials=num_trials,
                     hyperparameter_tune=hp_tune, hyperparameters=hyperparameters,
                     search_strategy=search_strategy)
```

We again demonstrate how to use the trained models to predict on the test data.

```{.python .input}
y_pred = predictor.predict(test_data_nolabel)
print("Predictions:  ", list(y_pred)[:5])
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=False)
```

Use the following to view a summary of what happened during fit. Now this command will show details of the hyperparameter-tuning process for each type of model:

```{.python .input}
results = predictor.fit_summary()
```

In the above example, the predictive performance may be poor because we specified very little training to ensure quick runtimes.  You can call `fit()` multiple times while modifying the above settings to better understand how these choices affect performance outcomes. For example: you can comment out the `train_data.head` command to train using a larger dataset, increase the `num_epochs` and `num_boost_round` hyperparameters, and increase the `time_limits` (which you should do for all code in these tutorials).  To see more detailed output during the execution of `fit()`, you can also pass in the argument: `verbosity = 3`.


## Model ensembling with stacking/bagging

Beyond hyperparameter-tuning with a correctly-specified evaluation metric, two other methods to boost predictive performance are [bagging and stack-ensembling](https://arxiv.org/abs/2003.06505).  You'll often see performance improve if you specify `num_bagging_folds` = 5-10, `stack_ensemble_levels` = 1-3 in the call to `fit()`, but this will increase training times and memory/disk usage.


```{.python .input}
predictor = task.fit(train_data=train_data, label=label_column, eval_metric=metric,
                     num_bagging_folds=5, stack_ensemble_levels=1,
                     hyperparameters = {'NN':{'num_epochs':5}, 'GBM':{'num_boost_round':100}}) # last 2 arguments are for quick demo, omit them in real applications
```

You should not provide `tuning_data` when stacking/bagging, and instead provide all your available data as `train_data` (which AutoGluon will split in more intellgent ways). Rather than manually searching for good bagging/stacking values yourself, AutoGluon will automatically select good values for you if you specify `auto_stack` instead:

```{.python .input}
output_directory = 'agModels-predictOccupation'  # folder where to store trained models

predictor = task.fit(train_data=train_data, label=label_column, eval_metric=metric,
                     auto_stack=True, output_directory=output_directory,
                     hyperparameters = {'NN':{'num_epochs':5}, 'GBM':{'num_boost_round':100}}, time_limits = 30) # last 2 arguments are for quick demo, omit them in real applications
```
Often stacking/bagging will produce superior accuracy than hyperparameter-tuning, but you may experiment with combining both techniques.


## Getting predictions (inference-time options)

Even if you've started a new Python session (possibly on a new machine) since last calling `fit()`, you can still load a previously trained predictor from disk:

```{.python .input}
predictor = task.load(output_directory)
```

Here, `output_directory` is the same folder previously passed to `fit()`, in which all the trained models have been saved.
You can train easily models on one machine and deploy them on another. Simply copy the `output_directory` folder to the new machine and specify its new path in `task.load()`.

We can make a prediction on an individual example rather than a full dataset:

```{.python .input}
datapoint = test_data_nolabel.iloc[[0]]  # Note: .iloc[0] won't work because it returns pandas Series instead of DataFrame
print(datapoint)
print(predictor.predict(datapoint))
```

To output predicted class probabilities instead of predicted classes, you can use:

```{.python .input}
class_probs = predictor.predict_proba(datapoint)
print(pd.DataFrame(class_probs, columns=predictor.class_labels))
```

By default, `predict()` and `predict_proba()` will utilize the model that AutoGluon thinks is most accurate, which is usually an ensemble of many individual models.
We can instead specify a particular model to use for predictions (e.g. to reduce inference latency).  Before deciding which model to use, let's evaluate all of the models AutoGluon has previously trained using our test data:

```{.python .input}
results = predictor.leaderboard(test_data)
```

Here's how to specify a particular model to use for prediction instead of AutoGluon's default model-choice:

```{.python .input}
i = 0  # index of model to use
model_to_use = predictor.get_model_names()[i]
model_pred = predictor.predict(datapoint, model=model_to_use)
print("Prediction from %s model: %s" % (model_to_use, model_pred))
```

We can easily access various information about the trained predictor or a particular model:
```{.python .input}
model_to_use = predictor.model_names[i]
specific_model = predictor._trainer.load_model(model_to_use)

# Objects defined below are dicts of various information (not printed here as they are quite large):
model_info = specific_model.get_info()
predictor_information = predictor.info()
```

The `predictor` also remembers what metric predictions should be evaluated with, which can be done with ground truth labels as follows:

```{.python .input}
y_pred = predictor.predict(test_data_nolabel)
predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
```

However, you must be careful here as certain metrics require predicted probabilities rather than classes.
Since the label columns remains in the `test_data` DataFrame, we can instead use the shorthand:

```{.python .input}
predictor.evaluate(test_data)
```

which will correctly select between `predict()` or `predict_proba()` depending on the evaluation metric.


## Accelerating inference

We describe multiple ways to reduce the time it takes for AutoGluon to produce predictions.

### Retaining all models in memory

By default, AutoGluon loads models into memory one at a time and only when they are needed for prediction. This strategy is robust for large stacked/bagged ensembles, but leads to slower prediction times. If you plan to repeatedly make predictions (e.g. on new datapoints one at a time rather than one large test dataset), you can first specify that all models should be loaded into memory as follows:

```{.python .input}
predictor._learner.persist_trainer() # load models into memory (still experimental, may fail for big ensembles)

num_test = 20
preds = np.array(['']*num_test, dtype='object')
for i in range(num_test):
    datapoint = test_data_nolabel.iloc[[i]]
    pred_numpy = predictor.predict(datapoint)
    preds[i] = pred_numpy[0]

perf = predictor.evaluate_predictions(y_test[:num_test], preds, auxiliary_metrics=True)
print("Predictions: ", preds)

predictor = task.load(output_directory) # reset predictor
```

### Collapsing bagged ensembles via refit_full()

For a ensemble predictor trained with bagging (as done above), recall there ~10 bagged copies of each individual model trained on different train/validation folds. We can collapse this bag of ~10 models into a single model that's fit to the full dataset, which can greatly reduce its memory/latency requirements. Below we refit such models for every model-type but you can alternatively do this for just a particular model-type by specifying the `model` argument of `refit_full()`.

```{.python .input}
refit_model_map = predictor.refit_full()
print("Name of each refit-full model corresponding to a previous bagged ensemble:")
print(refit_model_map)
predictor.leaderboard(test_data)
```
This adds the refit-full models to the leaderboard and we can opt to use any of them for prediction just like any other model. Note `pred_time_test` and `pred_time_val` list the time taken to produce predictions with each model (in seconds) on the test/validation data. Since the refit-full models were trained using all of the data, there is no internal validation score (`score_val`) available for them.

### Model distillation

While computationally-favorable, single individual models will usually have lower accuracy than weighted/stacked/bagged ensembles. [Model Distillation](https://arxiv.org/abs/2006.14284) offers one way the retain the computational benefits of a single model, while enjoying some of the accuracy-boost that comes with ensembling. The idea is to train the individual model (which we can call the student) to mimic the predictions of the full stack ensemble (the teacher). Like `refit_full()`, the `distill()` function will produce additional models we can opt to use for prediction.

```{.python .input}
student_models = predictor.distill(time_limits=30) # specify much longer time-limits in real applications
print(student_models)
preds_student = predictor.predict(test_data_nolabel, model=student_models[0])
print(f"predictions from {student_models[0]}:", preds_student)
predictor.leaderboard(test_data)
```

### Specifying presets or hyperparameters

Instead of trying to speed up a cumbersome trained model at prediction time, if you know inference latency or memory will be an issue at the outset, then you can adjust the training process accordingly to ensure `fit()` does not produce unwieldy models.

One option is to specify more lightweight `presets`:

```{.python .input}
presets = ['good_quality_faster_inference_only_refit', 'optimize_for_deployment']
predictor_light = task.fit(train_data=train_data, label=label_column, eval_metric=metric,
                           presets=presets, time_limits=30)
```

Another option is to specify more lightweight hyperparameters:

```{.python .input}
predictor_light = task.fit(train_data=train_data, label=label_column, eval_metric=metric,
                           hyperparameters='very_light', time_limits=30)
```

Here you can set `hyperparameters` to 'light','very_light', 'toy' to obtain progressively smaller (but less accurate) models and predictors. Advanced users may instead try manually specifying particular models' hyperparameters in order to make them faster/smaller.

Finally, you may also exclude specific unwieldy models from being trained at all. Below we exclude models that tend to be slower (K Nearest Neighbors, Neural Network, models with custom larger-than-default  hyperparameters):

```{.python .input}
excluded_model_types = ['KNN','NN','custom']
predictor_light = task.fit(train_data=train_data, label=label_column, eval_metric=metric,
                           excluded_model_types=excluded_model_types, time_limits=30)
```

If you encounter memory issues: try setting `excluded_model_types = ['KNN','RF','XT']`, and add `'ignore_text'` to your `presets` list if there happen to be text fields in your data.
If you encounter disk space issues, make sure to delete all `output_directory` folders from previous previous runs! These can eat up your free space if you call `fit()` many times. If you didn't specify `output_directory`, AutoGluon still automatically saved its models to a folder called: "AutogluonModels/ag-[TIMESTAMP]", where TIMESTAMP records when `fit()` was called, so make sure to also delete these folders if you run low on free space.





