# Forecasting Time-Series - In Depth Tutorial
:label:`sec_forecastingindepth`

This more advanced tutorial describes how you can exert greater control over AutoGluon's time-series modeling. As an example forecasting task, we again use the [Covid-19 dataset](https://www.kaggle.com/c/covid19-global-forecasting-week-4) previously described in the :ref:`sec_forecastingquick` tutorial.

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
