# Predicting Columns in a Table - In Depth
:label:`sec_tabularadvanced`

**Tip**: If you are new to AutoGluon, review :ref:`sec_tabularquick` to learn the basics of the AutoGluon API. To learn how to add your own custom models to the set that AutoGluon trains, tunes, and ensembles, review :ref:`sec_tabularcustommodel`.

This tutorial describes how you can exert greater control when using AutoGluon's `fit()` or `predict()`. Recall that to maximize predictive performance, you should first try `TabularPredictor()` and `fit()` with all default arguments.  Then, consider non-default arguments for `TabularPredictor(eval_metric=...)`, and `fit(presets=...)`.  Later, you can experiment with other arguments to fit() covered in this in-depth tutorial like `hyperparameter_tune_kwargs`, `hyperparameters`, `num_stack_levels`, `num_bag_folds`, `num_bag_sets`, etc.

Using the same census data table as in the :ref:`sec_tabularquick` tutorial, we'll now predict the `occupation` of an individual - a multiclass classification problem. Start by importing AutoGluon's TabularPredictor and TabularDataset, and loading the data.

```{.python .input}
from autogluon.tabular import TabularDataset, TabularPredictor

import numpy as np

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
print(train_data.head())

label = 'occupation'
print("Summary of occupation column: \n", train_data['occupation'].describe())

new_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
test_data = new_data[5000:].copy()  # this should be separate data in your applications
y_test = test_data[label]
test_data_nolabel = test_data.drop(columns=[label])  # delete label column
val_data = new_data[:5000].copy()

metric = 'accuracy' # we specify eval-metric just for demo (unnecessary as it's the default)
```

## Specifying hyperparameters and tuning them

We first demonstrate hyperparameter-tuning and how you can provide your own validation dataset that AutoGluon internally relies on to: tune hyperparameters, early-stop iterative training, and construct model ensembles. One reason you may specify validation data is when future test data will stem from a different distribution than training data (and your specified validation data is more representative of the future data that will likely be encountered).

 If you don't have a strong reason to provide your own validation dataset, we recommend you omit the `tuning_data` argument. This lets AutoGluon automatically select validation data from your provided training set (it uses smart strategies such as stratified sampling).  For greater control, you can specify the `holdout_frac` argument to tell AutoGluon what fraction of the provided training data to hold out for validation.

**Caution:** Since AutoGluon tunes internal knobs based on this validation data, performance estimates reported on this data may be over-optimistic. For unbiased performance estimates, you should always call `predict()` on a separate dataset (that was never passed to `fit()`), as we did in the previous **Quick-Start** tutorial. We also emphasize that most options specified in this tutorial are chosen to minimize runtime for the purposes of demonstration and you should select more reasonable values in order to obtain high-quality models.

`fit()` trains neural networks and various types of tree ensembles by default. You can specify various hyperparameter values for each type of model. For each hyperparameter, you can either specify a single fixed value, or a search space of values to consider during hyperparameter optimization. Hyperparameters which you do not specify are left at default settings chosen automatically by AutoGluon, which may be fixed values or search spaces.

```{.python .input}
import autogluon.core as ag

nn_options = {  # specifies non-default hyperparameter values for neural network models
    'num_epochs': 10,  # number of training epochs (controls training time of NN models)
    'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),  # learning rate used in training (real-valued hyperparameter searched on log-scale)
    'activation': ag.space.Categorical('relu', 'softrelu', 'tanh'),  # activation function used in NN (categorical hyperparameter, default = first entry)
    'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),  # dropout probability (real-valued hyperparameter)
}

gbm_options = {  # specifies non-default hyperparameter values for lightGBM gradient boosted trees
    'num_boost_round': 100,  # number of boosting rounds (controls training time of GBM models)
    'num_leaves': ag.space.Int(lower=26, upper=66, default=36),  # number of leaves in trees (integer hyperparameter)
}

hyperparameters = {  # hyperparameters of each model type
                   'GBM': gbm_options,
                   'NN_TORCH': nn_options,  # NOTE: comment this line out if you get errors on Mac OSX
                  }  # When these keys are missing from hyperparameters dict, no models of that type are trained

time_limit = 2*60  # train various models for ~2 min
num_trials = 5  # try at most 5 different hyperparameter configurations for each type of model
search_strategy = 'auto'  # to tune hyperparameters using random search routine with a local scheduler

hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
    'num_trials': num_trials,
    'scheduler' : 'local',
    'searcher': search_strategy,
}

predictor = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data, tuning_data=val_data, time_limit=time_limit,
    hyperparameters=hyperparameters, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)
```

We again demonstrate how to use the trained models to predict on the test data.

```{.python .input}
y_pred = predictor.predict(test_data_nolabel)
print("Predictions:  ", list(y_pred)[:5])
perf = predictor.evaluate(test_data, auxiliary_metrics=False)
```

Use the following to view a summary of what happened during `fit()`. Now this command will show details of the hyperparameter-tuning process for each type of model:

```{.python .input}
results = predictor.fit_summary(show_plot=True)
```

In the above example, the predictive performance may be poor because we specified very little training to ensure quick runtimes.  You can call `fit()` multiple times while modifying the above settings to better understand how these choices affect performance outcomes. For example: you can comment out the `train_data.head` command or increase `subsample_size` to train using a larger dataset, increase the `num_epochs` and `num_boost_round` hyperparameters, and increase the `time_limit` (which you should do for all code in these tutorials).  To see more detailed output during the execution of `fit()`, you can also pass in the argument: `verbosity = 3`.


## Model ensembling with stacking/bagging

Beyond hyperparameter-tuning with a correctly-specified evaluation metric, two other methods to boost predictive performance are [bagging and stack-ensembling](https://arxiv.org/abs/2003.06505).  You'll often see performance improve if you specify `num_bag_folds` = 5-10, `num_stack_levels` = 1-3 in the call to `fit()`, but this will increase training times and memory/disk usage.

```{.python .input}
predictor = TabularPredictor(label=label, eval_metric=metric).fit(train_data,
    num_bag_folds=5, num_bag_sets=1, num_stack_levels=1,
    hyperparameters = {'NN_TORCH': {'num_epochs': 2}, 'GBM': {'num_boost_round': 20}},  # last  argument is just for quick demo here, omit it in real applications
)
```

You should not provide `tuning_data` when stacking/bagging, and instead provide all your available data as `train_data` (which AutoGluon will split in more intellgent ways). `num_bag_sets` controls how many times the k-fold bagging process is repeated to further reduce variance (increasing this may further boost accuracy but will substantially increase training times, inference latency, and memory/disk usage). Rather than manually searching for good bagging/stacking values yourself, AutoGluon will automatically select good values for you if you specify `auto_stack` instead:

```{.python .input}
save_path = 'agModels-predictOccupation'  # folder where to store trained models

predictor = TabularPredictor(label=label, eval_metric=metric, path=save_path).fit(
    train_data, auto_stack=True,
    time_limit=30, hyperparameters={'NN_TORCH': {'num_epochs': 2}, 'GBM': {'num_boost_round': 20}}  # last 2 arguments are for quick demo, omit them in real applications
)
```

Often stacking/bagging will produce superior accuracy than hyperparameter-tuning, but you may try combining both techniques (note: specifying `presets='best_quality'` in `fit()` simply sets `auto_stack=True`).


## Prediction options (inference)

Even if you've started a new Python session since last calling `fit()`, you can still load a previously trained predictor from disk:

```{.python .input}
predictor = TabularPredictor.load(save_path)  # `predictor.path` is another way to get the relative path needed to later load predictor.
```

Above `save_path` is the same folder previously passed to `TabularPredictor`, in which all the trained models have been saved. You can train easily models on one machine and deploy them on another. Simply copy the `save_path` folder to the new machine and specify its new path in `TabularPredictor.load()`.

To find out the required feature columns to make predictions, call `predictor.features()`:

```{.python .input}
predictor.features()
```

We can make a prediction on an individual example rather than a full dataset:

```{.python .input}
datapoint = test_data_nolabel.iloc[[0]]  # Note: .iloc[0] won't work because it returns pandas Series instead of DataFrame
print(datapoint)
predictor.predict(datapoint)
```

To output predicted class probabilities instead of predicted classes, you can use:

```{.python .input}
predictor.predict_proba(datapoint)  # returns a DataFrame that shows which probability corresponds to which class
```

By default, `predict()` and `predict_proba()` will utilize the model that AutoGluon thinks is most accurate, which is usually an ensemble of many individual models. Here's how to see which model this is:

```{.python .input}
predictor.get_model_best()
```

We can instead specify a particular model to use for predictions (e.g. to reduce inference latency). Note that a 'model' in AutoGluon may refer to, for example, a single Neural Network, a bagged ensemble of many Neural Network copies trained on different training/validation splits, a weighted ensemble that aggregates the predictions of many other models, or a stacker model that operates on predictions output by other models. This is akin to viewing a Random Forest as one 'model' when it is in fact an ensemble of many decision trees.


Before deciding which model to use, let's evaluate all of the models AutoGluon has previously trained on our test data:

```{.python .input}
predictor.leaderboard(test_data, silent=True)
```

The leaderboard shows each model's predictive performance on the test data (`score_test`) and validation data (`score_val`), as well as the time required to: produce predictions for the test data (`pred_time_val`), produce predictions on the validation data (`pred_time_val`), and train only this model (`fit_time`). Below, we show that a leaderboard can be produced without new data (just uses the data previously reserved for validation inside `fit`) and can display extra information about each model:

```{.python .input}
predictor.leaderboard(extra_info=True, silent=True)
```

The expanded leaderboard shows properties like how many features are used by each model (`num_features`), which other models are ancestors whose predictions are required inputs for each model (`ancestors`), and how much memory each model and all its ancestors would occupy if simultaneously persisted (`memory_size_w_ancestors`). See the [leaderboard documentation](../../api/autogluon.predictor.html#autogluon.tabular.TabularPredictor.leaderboard) for full details.

To show scores for other metrics, you can specify the `extra_metrics` argument when passing in `test_data`:

```{.python .input}
predictor.leaderboard(test_data, extra_metrics=['accuracy', 'balanced_accuracy', 'log_loss'], silent=True)
```

Notice that `log_loss` scores are negative.
This is because metrics in AutoGluon are always shown in `higher_is_better` form.
This means that metrics such as `log_loss` and `root_mean_squared_error` will have their signs FLIPPED, and values will be negative.
This is necessary to avoid the user needing to know the metric to understand if higher is better when looking at leaderboard.

One additional caveat: It is possible that `log_loss` values can be `-inf` when computed via `extra_metrics`.
This is because the models were not optimized with `log_loss` in mind during training and
may have prediction probabilities giving a class `0` (particularly common with K-Nearest-Neighbors models).
Because `log_loss` gives infinite error when the correct class was given `0` probability, this results in a score of `-inf`.
It is therefore recommended that `log_loss` should not be used as a secondary metric to determine model quality.
Either use `log_loss` as the `eval_metric` or avoid it altogether.

Here's how to specify a particular model to use for prediction instead of AutoGluon's default model-choice:

```{.python .input}
i = 0  # index of model to use
model_to_use = predictor.get_model_names()[i]
model_pred = predictor.predict(datapoint, model=model_to_use)
print("Prediction from %s model: %s" % (model_to_use, model_pred.iloc[0]))
```

We can easily access various information about the trained predictor or a particular model:
```{.python .input}
all_models = predictor.get_model_names()
model_to_use = all_models[i]
specific_model = predictor._trainer.load_model(model_to_use)

# Objects defined below are dicts of various information (not printed here as they are quite large):
model_info = specific_model.get_info()
predictor_information = predictor.info()
```

The `predictor` also remembers what metric predictions should be evaluated with, which can be done with ground truth labels as follows:

```{.python .input}
y_pred_proba = predictor.predict_proba(test_data_nolabel)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred_proba)
```

Since the label columns remains in the `test_data` DataFrame, we can instead use the shorthand:

```{.python .input}
perf = predictor.evaluate(test_data)
```

## Interpretability (feature importance)

To better understand our trained predictor, we can estimate the overall importance of each feature:

```{.python .input}
predictor.feature_importance(test_data)
```

Computed via [permutation-shuffling](https://explained.ai/rf-importance/), these feature importance scores quantify the drop in predictive performance (of the already trained predictor) when one column's values are randomly shuffled across rows. The top features in this list contribute most to AutoGluon's accuracy (for predicting when/if a patient will be readmitted to the hospital). Features with non-positive importance score hardly contribute to the predictor's accuracy, or may even be actively harmful to include in the data (consider removing these features from your data and calling `fit` again). These scores facilitate interpretability of the predictor's global behavior (which features it relies on for *all* predictions). To get [local explanations](https://christophm.github.io/interpretable-ml-book/taxonomy-of-interpretability-methods.html) regarding which features influence a *particular* prediction, check out the [example notebooks](https://github.com/awslabs/autogluon/tree/master/examples/tabular/interpret) for explaining particular AutoGluon predictions using [Shapely values](https://github.com/slundberg/shap/).

## Accelerating inference

We describe multiple ways to reduce the time it takes for AutoGluon to produce predictions.

Before providing code examples, it is important to understand that
there are several ways to accelerate inference in AutoGluon. The table below lists the options in order of priority.

| Optimization                      |                                          Inference Speedup                                           | Cost                              | Notes                                                                                                                                                                                  |
|:----------------------------------|:----------------------------------------------------------------------------------------------------:|:----------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| refit_full                        |                             At least 8x+, up to 160x (requires bagging)                              | -Quality, +FitTime                | Only provides speedup with bagging enabled.                                                                                                                                            |
| persist_models                    |                                    Up to 10x in online-inference                                     | ++MemoryUsage                     | If memory is not sufficient to persist model, speedup is not gained. Speedup is most effective in online-inference and is not relevant in batch inference.                             |
| infer_limit                       |                                       Configurable, ~up to 50x                                       | -Quality (Relative to speedup)    | If bagging is enabled, always use refit_full if using infer_limit.                                                                                                                     |
| distill                           |             ~Equals combined speedup of refit_full and infer_limit set to extreme values             | --Quality, ++FitTime              | Not compatible with refit_full and infer_limit.                                                                                                                                        |
| feature pruning                   |               Typically at most 1.5x. More if willing to lower quality significantly.                | -Quality?, ++FitTime              | Dependent on the existence of unimportant features in data. Call `predictor.feature_importance(test_data)` to gauge which features could be removed.                                   |
| use faster hardware               |                       Usually at most 3x. Depends on hardware (ignoring GPU).                        | +Hardware                         | As an example, an EC2 c6i.2xlarge is ~1.6x faster than an m5.2xlarge for a similar price. Laptops in particular might be slow compared to cloud instances.                             |
| manual hyperparameters adjustment |                    Usually at most 2x assuming infer_limit is already specified.                     | ---Quality?, +++UserMLExpertise   | Can be very complicated and is not recommended. Potential ways to get speedups this way is to reduce the number of trees in LightGBM, XGBoost, CatBoost, RandomForest, and ExtraTrees. |
| manual data preprocessing         | Usually at most 1.2x assuming all other optimizations are specified and setting is online-inference. | ++++UserMLExpertise, ++++UserCode | Only relevant for online-inference. This is not recommended as AutoGluon's default preprocessing is highly optimized.                                                                  |

If bagging is enabled (num_bag_folds>0 or num_stack_levels>0 or using 'best_quality' preset), the order of inference optimizations should be:
1. refit_full
2. persist_models
3. infer_limit

If bagging is not enabled (num_bag_folds=0, num_stack_levels=0), the order of inference optimizations should be:
1. persist_models
2. infer_limit

If following these recommendations does not lead to a sufficiently fast model, you may consider the more advanced options in the table.

### Keeping models in memory

By default, AutoGluon loads models into memory one at a time and only when they are needed for prediction. This strategy is robust for large stacked/bagged ensembles, but leads to slower prediction times. If you plan to repeatedly make predictions (e.g. on new datapoints one at a time rather than one large test dataset), you can first specify that all models required for inference should be loaded into memory as follows:

```{.python .input}
predictor.persist_models()

num_test = 20
preds = np.array(['']*num_test, dtype='object')
for i in range(num_test):
    datapoint = test_data_nolabel.iloc[[i]]
    pred_numpy = predictor.predict(datapoint, as_pandas=False)
    preds[i] = pred_numpy[0]

perf = predictor.evaluate_predictions(y_test[:num_test], preds, auxiliary_metrics=True)
print("Predictions: ", preds)

predictor.unpersist_models()  # free memory by clearing models, future predict() calls will load models from disk
```

You can alternatively specify a particular model to persist via the `models` argument of `persist_models()`, or simply set `models='all'` to simultaneously load every single model that was trained during `fit`.

### Inference speed as a fit constraint

If you know your latency constraint prior to fitting the predictor, you can specify it explicitly as a fit argument.
AutoGluon will then automatically train models in a fashion that attempts to satisfy the constraint.

This constraint has two components: `infer_limit` and `infer_limit_batch_size`:  
- `infer_limit` is the time in seconds to predict 1 row of data.
For example, `infer_limit=0.05` means 50 ms per row of data,
or 20 rows / second throughput.
- `infer_limit_batch_size` is the amount of rows passed at once to predict when calculating per-row speed.
This is very important because `infer_limit_batch_size=1` (online-inference) is highly suboptimal as
various operations have a fixed cost overhead regardless of data size. If you can pass your test data in bulk,
you should specify `infer_limit_batch_size=10000`.

```{.python .input}
# At most 0.05 ms per row (20000 rows per second throughput)
infer_limit = 0.00005
# adhere to infer_limit with batches of size 10000 (batch-inference, easier to satisfy infer_limit)
infer_limit_batch_size = 10000
# adhere to infer_limit with batches of size 1 (online-inference, much harder to satisfy infer_limit)
# infer_limit_batch_size = 1  # Note that infer_limit<0.02 when infer_limit_batch_size=1 can be difficult to satisfy.
predictor_infer_limit = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data=train_data,
    time_limit=30,
    infer_limit=0.00005,  
    # adhere to infer_limit with batches of size 10000 (batch-inference, easier to satisfy infer_limit)
    infer_limit_batch_size=infer_limit_batch_size,
)

# NOTE: If bagging was enabled, it is important to call refit_full at this stage.
#  infer_limit assumes that the user will call refit_full after fit.
# predictor_infer_limit.refit_full()

# NOTE: To align with inference speed calculated during fit, models must be persisted.
predictor_infer_limit.persist_models()
# Below is an optimized version that only persists the minimum required models for prediction.
# predictor_infer_limit.persist_models('best')

predictor.leaderboard(silent=True)
```

Now we can test the inference speed of the final model and check if it satisfies the inference constraints.

```{.python .input}
test_data_batch = test_data.sample(infer_limit_batch_size, replace=True, ignore_index=True)

import time
time_start = time.time()
predictor_infer_limit.predict(test_data_batch)
time_end = time.time()

infer_time_per_row = (time_end - time_start) / len(test_data_batch)
rows_per_second = 1 / infer_time_per_row
infer_time_per_row_ratio = infer_time_per_row / infer_limit
is_constraint_satisfied = infer_time_per_row_ratio <= 1

print(f'Model is able to predict {round(rows_per_second, 1)} rows per second. (User-specified Throughput = {1 / infer_limit})')
print(f'Model uses {round(infer_time_per_row_ratio * 100, 1)}% of infer_limit time per row.')
print(f'Model satisfies inference constraint: {is_constraint_satisfied}')
```

### Using smaller ensemble or faster model for prediction

Without having to retrain any models, one can construct alternative ensembles that aggregate individual models' predictions with different weighting schemes. These ensembles become smaller (and hence faster for prediction) if they assign nonzero weight to less models. You can produce a wide variety of ensembles with different accuracy-speed tradeoffs like this:

```{.python .input}
additional_ensembles = predictor.fit_weighted_ensemble(expand_pareto_frontier=True)
print("Alternative ensembles you can use for prediction:", additional_ensembles)

predictor.leaderboard(only_pareto_frontier=True, silent=True)
```

The resulting leaderboard will contain the most accurate model for a given inference-latency. You can select whichever model exhibits acceptable latency from the leaderboard and use it for prediction.

```{.python .input}
model_for_prediction = additional_ensembles[0]
predictions = predictor.predict(test_data, model=model_for_prediction)
predictor.delete_models(models_to_delete=additional_ensembles, dry_run=False)  # delete these extra models so they don't affect rest of tutorial
```

### Collapsing bagged ensembles via refit_full

For an ensemble predictor trained with bagging (as done above), recall there are ~10 bagged copies of each individual model trained on different train/validation folds. We can collapse this bag of ~10 models into a single model that's fit to the full dataset, which can greatly reduce its memory/latency requirements (but may also reduce accuracy). Below we refit such a model for each original model but you can alternatively do this for just a particular model by specifying the `model` argument of `refit_full()`.

```{.python .input}
refit_model_map = predictor.refit_full()
print("Name of each refit-full model corresponding to a previous bagged ensemble:")
print(refit_model_map)
predictor.leaderboard(test_data, silent=True)
```
This adds the refit-full models to the leaderboard and we can opt to use any of them for prediction just like any other model. Note `pred_time_test` and `pred_time_val` list the time taken to produce predictions with each model (in seconds) on the test/validation data. Since the refit-full models were trained using all of the data, there is no internal validation score (`score_val`) available for them. You can also call `refit_full()` with non-bagged models to refit the same models to your full dataset (there won't be memory/latency gains in this case but test accuracy may improve).

### Model distillation

While computationally-favorable, single individual models will usually have lower accuracy than weighted/stacked/bagged ensembles. [Model Distillation](https://arxiv.org/abs/2006.14284) offers one way to retain the computational benefits of a single model, while enjoying some of the accuracy-boost that comes with ensembling. The idea is to train the individual model (which we can call the student) to mimic the predictions of the full stack ensemble (the teacher). Like `refit_full()`, the `distill()` function will produce additional models we can opt to use for prediction.

```{.python .input}
student_models = predictor.distill(time_limit=30)  # specify much longer time limit in real applications
print(student_models)
preds_student = predictor.predict(test_data_nolabel, model=student_models[0])
print(f"predictions from {student_models[0]}:", list(preds_student)[:5])
predictor.leaderboard(test_data)
```

### Faster presets or hyperparameters

Instead of trying to speed up a cumbersome trained model at prediction time, if you know inference latency or memory will be an issue at the outset, then you can adjust the training process accordingly to ensure `fit()` does not produce unwieldy models.

One option is to specify more lightweight `presets`:

```{.python .input}
presets = ['good_quality_faster_inference_only_refit', 'optimize_for_deployment']
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(train_data, presets=presets, time_limit=30)
```

Another option is to specify more lightweight hyperparameters:

```{.python .input}
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(train_data, hyperparameters='very_light', time_limit=30)
```

Here you can set `hyperparameters` to either 'light', 'very_light', or 'toy' to obtain progressively smaller (but less accurate) models and predictors. Advanced users may instead try manually specifying particular models' hyperparameters in order to make them faster/smaller.

Finally, you may also exclude specific unwieldy models from being trained at all. Below we exclude models that tend to be slower (K Nearest Neighbors, Neural Network, models with custom larger-than-default  hyperparameters):

```{.python .input}
excluded_model_types = ['KNN', 'NN_TORCH', 'custom']
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(train_data, excluded_model_types=excluded_model_types, time_limit=30)
```


## If you encounter memory issues

To reduce memory usage during training, you may try each of the following strategies individually or combinations of them (these may harm accuracy):

- In `fit()`, set `num_bag_sets = 1` (can also try values greater than 1 to harm accuracy less).

- In `fit()`, set `excluded_model_types = ['KNN', 'XT' ,'RF']` (or some subset of these models).

- Try different `presets` in `fit()`.

- In `fit()`, set `hyperparameters = 'light'` or `hyperparameters = 'very_light'`.

- Text fields in your table require substantial memory for N-gram featurization. To mitigate this in `fit()`, you can either: (1) add `'ignore_text'` to your `presets` list (to ignore text features), or (2) specify the argument:

```
from sklearn.feature_extraction.text import CountVectorizer
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
feature_generator = AutoMLPipelineFeatureGenerator(vectorizer=CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=MAX_NGRAM, dtype=np.uint8))
```

for example using `MAX_NGRAM = 1000` (try various values under 10000 to reduce the number of N-gram features used to represent each text field)

In addition to reducing memory usage, many of the above strategies can also be used to reduce training times.

To reduce memory usage during inference:

- If trying to produce predictions for a large test dataset, break the test data into smaller chunks as demonstrated in :ref:`sec_faq`.

- If models have been previously persisted in memory but inference-speed is not a major concern, call `predictor.unpersist_models()`.

- If models have been previously persisted in memory, bagging was used in `fit()`, and inference-speed is a concern: call `predictor.refit_full()` and use one of the refit-full models for prediction (ensure this is the only model persisted in memory).



## If you encounter disk space issues

To reduce disk usage, you may try each of the following strategies individually or combinations of them:

- Make sure to delete all `predictor.path` folders from previous `fit()` runs! These can eat up your free space if you call `fit()` many times. If you didn't specify `path`, AutoGluon still automatically saved its models to a folder called: "AutogluonModels/ag-[TIMESTAMP]", where TIMESTAMP records when `fit()` was called, so make sure to also delete these folders if you run low on free space.

- Call `predictor.save_space()` to delete auxiliary files produced during `fit()`.

- Call `predictor.delete_models(models_to_keep='best', dry_run=False)` if you only intend to use this predictor for inference going forward (will delete files required for non-prediction-related functionality like `fit_summary`).

- In `fit()`, you can add `'optimize_for_deployment'` to the `presets` list, which will automatically invoke the previous two strategies after training.

- Most of the above strategies to reduce memory usage will also reduce disk usage (but may harm accuracy).


## References

The following paper describes how AutoGluon internally operates on tabular data:

Erickson et al. [AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/abs/2003.06505). *Arxiv*, 2020.
