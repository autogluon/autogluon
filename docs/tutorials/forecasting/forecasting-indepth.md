# Forecasting Time-Series - In Depth Tutorial
:label:`sec_forecastingindepth`

This more advanced tutorial describes how you can exert greater control over AutoGluon's time-series modeling. As an example forecasting task, we again use the COVID-19 dataset previously described in :ref:`sec_forecastingquick`.

```{.python .input}
from autogluon.forecasting import ForecastingPredictor
from autogluon.forecasting import TabularDataset

train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")

save_path = "agModels-covidforecast"
eval_metric = "mean_wQuantileLoss"  # just for demonstration, this is already the default evaluation metric
```


## Specifying hyperparameters and tuning them

While AutoGluon-Forecasting will automatically tune certain hyperparameters of time-series models depending on the `presets` setting, you can manually control the hyperparameter optimization (HPO) process. The `presets` argument of `predictor.fit()` will automatically determine particular search spaces to consider for certain hyperparameter values, as well as how many HPO trials to run when searching for the best value in the chosen hyperparameter search space. Instead of specifying `presets`, you can manually specify all of these items yourself. Below we demonstrate how to tune the `context_length` hyperparameter for just the MQCNN model.

```{.python .input}
import autogluon.core as ag
from gluonts.mx.distribution.neg_binomial import NegativeBinomialOutput

context_search_space = ag.Int(75, 100)  # integer values spanning the listed range
epochs = 2  # small value used for quick demo, omit this or use much larger value in real applications!
num_batches_per_epoch = 5  # small value used for quick demo, omit this or use larger value in real applications!
num_hpo_trials = 2  # small value used for quick demo, use much larger value in real applications!

mqcnn_params = {
    "context_length": context_search_space,
    "epochs": epochs,
    "num_batches_per_epoch": num_batches_per_epoch,
}
deepar_params = {
    "context_length": context_search_space,
    "epochs": epochs,
    "num_batches_per_epoch": num_batches_per_epoch,
    "distr_output": NegativeBinomialOutput(),
}

predictor = ForecastingPredictor(path=save_path, eval_metric=eval_metric).fit(
    train_data, prediction_length=19, quantiles=[0.1, 0.5, 0.9],
    index_column="name", target_column="ConfirmedCases", time_column="Date",
    hyperparameter_tune_kwargs={'scheduler': 'local', 'searcher': 'bayesopt', 'num_trials': num_hpo_trials},
    hyperparameters={"MQCNN": mqcnn_params, "DeepAR": deepar_params}
)
```

To ensure quick runtimes, we specified that only 2 HPO trials should be run for tuning each model's hyperparameters, which is too few for real applications. We specified that HPO should be performed via a Bayesian optimization `searcher` with HPO trials to evaluate candidate hyperparameter configurations executed via a local sequential job `scheduler`. See the AutoGluon Searcher/Scheduler documentation/tutorials for more details.

Above we set the `epochs`, `num_batches_per_epoch`, and `distr_output` hyperparameters to fixed values. You are allowed to set some hyperparameters to search spaces and others to fixed values. Any hyperparameters you do not specify values or search spaces for will be left at their default values. AutoGluon will **only** train those models which appear as keys in the `hyperparameters` dict argument passed into `fit()`, so in this case only the MQCNN and DeepAR models are trained. Refer to the [GluonTS documentation](https://ts.gluon.ai/api/gluonts/gluonts.model.html) for individual GluonTS models to see all of the hyperparameters you may specify for them.


## Viewing additional information

We can view a summary of the HPO process, which will show the validation score achieved in each HPO trial as well as which hyperparameter configuration was evaluated in the corresponding trial:

```{.python .input}
predictor.fit_summary()
```

The `'best_config'` field in this summary indicates the hyperparameter configuration that performed best for each model. We can alternatively use the leaderboard to view the performance of each evaluated model/hyperparameter configuration:

```{.python .input}
predictor.leaderboard()
```

Here is yet another way to see which model AutoGluon believes to be the best (based on validation score), which is the model automatically used for prediction by default:

```{.python .input}
predictor._trainer.get_model_best()
```

We can also view information about any model AutoGluon has trained:

```{.python .input}
models_trained = predictor._trainer.get_model_names_all()
specific_model = predictor._trainer.load_model(models_trained[0])
specific_model.get_info()
```


## Evaluation

Given some more recent held-out test data, here's how to just evaluate the default model AutoGluon uses for forecasting without evaluating all of the other models as in `leaderboard()`:

```{.python .input}
test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")
predictor.evaluate(test_data)  # to evaluate specific model, can also specify optional argument: model
```

The ground truth time-series targets are often not available when we produce forecasts and only become available later in the future.
In such a workflow, we may first produce predictions using AutoGluon, and then later evaluate them without having to recompute the predictions:

```{.python .input}
predictions = predictor.predict(train_data)  # before test data have been observed

predictor = ForecastingPredictor.load(save_path)  # reload predictor in future after test data are observed
# reformatted_test_data = ForecastingPredictor.evaluation_format(test_data, train_data)  # TODO
# ForecastingPredictor.evaluate_predictions(forecasts=predictions, targets=test_data, eval_metric=predictor.eval_metric)  # TODO
```


## Static features

In some forecasting problems involving multiple time-series, each individual time-series may be associated with some static features that do not change over time. For example, if forecasting demand for products over time, each product may be associated with an item  category (categorical static feature) and an item vector embedding from a recommender system (numeric static features).
AutoGluon allows you to provide such static features such that its models will condition their predictions upon them:

```{.python .input}
static_features = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries"
                                 "/toy_static_features.csv")
static_features.head()
```

Note that each unique value of `index_column` in our time series data must be represented as a row in `static_features` (in a column whose name matches the `index_column`) that contains the feature values corresponding to this individual series. AutoGluon can automatically infer which static features are categorical vs. numeric when they are passed into `fit()`:


```{.python .input}
predictor = ForecastingPredictor(path=save_path, eval_metric=eval_metric).fit(
    train_data, static_features=static_features, prediction_length=19, quantiles=[0.1, 0.5, 0.9],
    index_column="name", target_column="ConfirmedCases", time_column="Date",
    hyperparameters={"MQCNN": {'epochs': epochs}}
)
```

If you provided static features to `fit()`, then the static features must be also provided when using `leaderboard()`, `evaluate()`, or `predict()`:

```{.python .input}
predictor.leaderboard(test_data, static_features=static_features)
```

```{.python .input}
predictions = predictor.predict(test_data, static_features=static_features)
print(predictions["Afghanistan_"])
```
