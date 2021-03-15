This notebook introduce more advanced techniques in Forecasting tasks.

Similar to the quick start tutourial, we will do forecasting related to the COV19 dataset.

```{.python .input}
from autogluon.forecasting import ForecastingPredictor
from autogluon.forecasting import TabularDataset

train_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/train.csv")
test_data = TabularDataset("https://autogluon.s3-us-west-2.amazonaws.com/datasets/CovidTimeSeries/test.csv")

save_path = "AutogluonModels/user/"
# uncessary as it's the default metric
eval_metric = "mean_wQuantileLoss"
# uncessary as it's the default searcher
searcher_type = "random"
```

# Specify hyperparameters and tuning them

We can do hyperparameter tuning using autogluon, and here the variable "context_length" is the one that we would like to tune.

```{.python .input}
import autogluon.core as ag
from gluonts.mx.distribution.neg_binomial import NegativeBinomialOutput


mqcnn_params = {
    "context_length": ag.Int(75, 100),
    "num_batches_per_epoch": 32,
    "epochs": 5
}

deepar_params = {
    "context_length": ag.Int(75, 100),
    "num_batches_per_epoch": 32,
    "distr_output": NegativeBinomialOutput(),
    "epochs": 5
}

sff_params = {
    "context_length": ag.Int(75, 100),
    "num_batches_per_epoch": 32,
    "epochs": 5
}

predictor = ForecastingPredictor(path=save_path, eval_metric=eval_metric).fit(train_data,
                                                                              prediction_length=19,
                                                                              index_column="name",
                                                                              target_column="ConfirmedCases",
                                                                              time_column="Date",
                                                                              hyperparameter_tune=True,
                                                                              quantiles=[0.1, 0.5, 0.9],
                                                                              refit_full=True,
                                                                              keep_only_best=True,
                                                                              set_best_to_refit_full=True,
                                                                              search_strategy=searcher_type,
                                                                              hyperparameters={
                                                                                  "MQCNN": mqcnn_params,
                                                                                  "DeepAR": deepar_params,
                                                                                  "SFF": sff_params,
                                                                              },
                                                                              num_trials=2
                                                                              )
```

```{.python .input}
predictor.fit_summary()
```

# Evaluation and Predictions

We again demonstrate how to use the trained models to predict on the test data.

To see the performance of each model on test data, we can use the leaderboard() method.

```{.python .input}
predictor.leaderboard(test_data)
```

By default, the predictor will use the best model on the validation set to do the prediction. The prediction results we get is a dictionary whose key is each time series's index and value is the corresponding prediction dataframe.

```{.python .input}
predictions = predictor.predict(test_data)
predictions["Afghanistan_"]
```

To see which model is the best, we can call the get_model_best() method for trainer.

```{.python .input}
predictor._trainer.get_model_best()
```

Besides using the best model on validation data, we can also specify which model we want to use for predictions.

```{.python .input}
model_trained = predictor._trainer.get_model_names_all()
specific_model = predictor._trainer.load_model(model_trained[0])
specific_model.get_info()
```

```{.python .input}
specific_predictions = predictor.predict(test_data, model=specific_model)
specific_predictions["Afghanistan_"]
```
