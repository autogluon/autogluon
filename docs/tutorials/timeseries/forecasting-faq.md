# AutoGluon Time Series FAQ

## What forecasting tasks can AutoGluon be used for?
AutoGluon can generate **probabilistic** multi-step-ahead forecasts for one or multiple **univariate** time series.
For example, you can use AutoGluon to forecast daily sales of multiple products over the next month.

AutoGluon also supports additional information, such as time-independent static features (e.g., location of the store)
and time-dependent covariates (e.g., price of the product each day).
See the [In Depth Tutorial](forecasting-indepth.ipynb) for more details.

## How can I get the most accurate forecasts?
To maximize the forecast accuracy, set the `predictor.fit()` argument `presets="best_quality"` or `presets="high_quality"` and provide a high `time_limit`.
Training usually takes up to 15-60 minutes for small and medium-sized datasets.
Large datasets with more than 5M rows may require up to 4-6 hours of training.

## How should I choose the evaluation metric?
See ["How should I choose the `eval_metric`?" in the In Depth Tutorial](forecasting-indepth.ipynb)

The `eval_metric` parameter allows you to control the metric optimized by

Definition of the metrics can be found in the [`autogluon.timeseries.evaluator`](https://github.com/autogluon/autogluon/blob/master/timeseries/src/autogluon/timeseries/evaluator.py) module.
An overview of different evaluation metrics is also available
[here](https://docs.aws.amazon.com/forecast/latest/dg/metrics.html).


## Are there any restrictions on the data that I can pass to TimeSeriesPredictor?
See ["What data can I pass to `TimeSeriesPredictor`?" in the In Depth Tutorial](forecasting-indepth.ipynb)

Time series in the dataset can have different lengths.

All time series in the dataset


All time series must be regularly sampled (e.g., measured every hour) and contain no missing values.
If your data is irregularly sampled or contains missing values, have a look at DDDDD.


## Can I use GPUs for model training?

Yes! All deep learning models used by `autogluon.timeseries` support GPU training.
The models will be automatically trained on a GPU if (1) your machine has a GPU and (2) you installed a PyTorch version with CUDA support.
Multi-GPU training is not yet supported.


## What machine is best for running AutoGluon TimeSeries?
AutoGluon can be run on any machine including your laptop.
It is not necessary to use a GPU to train `TimeSeriesPredictor`, so CPU machines are fine.
Using a machine with more CPU cores and more RAM will lead to faster training and allow you to quickly generate forecasts for larger datasets.
For example if using AWS instances for Tabular: we recommend [M6 instances](https://aws.amazon.com/ec2/instance-types/m6i/) instances, where a `m6i.24xlarge` machine should be able to handle most datasets.


## Issues not addressed here
First search if your issue is addressed in the [tutorials](index.md),
[documentation](../../api/autogluon.timeseries.TimeSeriesPredictor.rst), or [Github issues](https://github.com/autogluon/autogluon/issues)
(search both Closed and Open issues).
If it is not there, please open a [new Github Issue](https://github.com/autogluon/autogluon/issues/new) and
clearly state your issue and clarify how it relates to the module.

If you have a bug, please include: your code (ideally set `verbosity=4` which will print out more details), the
output printed during the code execution, and information about your operating system, Python version, and
installed packages (output of `pip freeze`).
Many user issues stem from incorrectly formatted data, so please describe your data as clearly as possible.
