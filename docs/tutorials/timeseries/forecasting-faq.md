# AutoGluon Time Series FAQ

## What forecasting tasks can AutoGluon be used for?
AutoGluon can generate **probabilistic** multi-step-ahead forecasts for one or multiple **univariate** time series.
For example, you can use AutoGluon to forecast daily sales of multiple products over the next month.
This setting should not be confused with multivariate time series forecasting (see [this discussion](https://stats.stackexchange.com/a/365394/134754) about the difference between multivariate and multiple univariate time series).

AutoGluon also supports additional information, such as time-independent static features (e.g., location of the store)
and time-dependent covariates (e.g., price of the product each day).
See the [In Depth Tutorial](forecasting-indepth.ipynb) for more details.

Currently, AutoGluon does not support features such as hierarchical forecasting and forecast explainability.

## How can I get the most accurate forecasts?
To maximize the forecast accuracy, set `presets="best_quality"` and provide a high `time_limit` when calling {py:meth}`~autogluon.timeseries.TimeSeriesPredictor.fit()`.

You can typically increase the forecast accuracy even further by increasing `num_val_windows` to 3-5 when calling `predictor.fit()`, but this will require an even longer training time. To speed up training when using multiple validation windows, you can increase the `refit_every_n_windows` argument to `predictor.fit()`.

## How should I choose the evaluation metric?
See [Evaluation Metrics](forecasting-metrics.md).

## Are there any restrictions on the data that I can pass to TimeSeriesPredictor?
See section "What data format is expected by `TimeSeriesPredictor`?" in the [In Depth Tutorial](forecasting-indepth.ipynb).


## Can I use GPUs for model training?
Yes! All deep learning models used by `autogluon.timeseries` support GPU training.
The models will be automatically trained on a GPU if (1) your machine has a GPU and (2) you installed a PyTorch version with CUDA support.
Multi-GPU training is not yet supported.


## What machine is best for running AutoGluon TimeSeries?
AutoGluon can be run on any machine including your laptop.
Having multiple CPU cores makes training faster for most forecasting models, and deep learning models additionally benefit from a GPU.
When using AWS, we recommend [G4](https://aws.amazon.com/ec2/instance-types/g4/) or [G5](https://aws.amazon.com/ec2/instance-types/g5/) instances with a single GPU and at least 16 CPU cores for fastest training on large datasets.

Machines without a GPU but with a large number of CPU cores are also well suited for training the `TimeSeriesPredictor`.
Among CPU-only instances on AWS, we recommend [M6](https://aws.amazon.com/ec2/instance-types/m6i/) instances, where a `m6i.24xlarge` machine should be able to handle most datasets.


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
