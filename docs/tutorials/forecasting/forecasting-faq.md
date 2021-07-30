# FAQ
:label:`sec_forecastingfaq`


### Where can I find more information about the models/metrics?

Most of the models and metrics are implemented via the [GluonTS package](https://ts.gluon.ai/), so please refer to their [documentation](https://ts.gluon.ai/api/gluonts/gluonts.html), [github](https://github.com/awslabs/gluon-ts), and paper:

[GluonTS: Probabilistic and Neural Time Series Modeling in Python](https://www.jmlr.org/papers/v21/19-820.html)

A description of certain evaluation metrics is available [here](https://docs.aws.amazon.com/forecast/latest/dg/metrics.html#metrics-wQL).


### How can I get the most accurate forecast predictions?

Generally setting the `predictor.fit()` argument `presets="best_quality"` will result in high accuracy. Alternative options include manually specifying hyperparameter search spaces for certain models and manually increasing the number of hyperparameter optimization trials, as demonstrated in the "In Depth" Tutorial.


### What are common data formatting issues?

Each (index, time) combination must only be associated with a single target value in your dataset, you cannot include multiple observations of the same time-series at the same time point. The time values in `time_column` must be in a valid [pandas time series format](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html) and must be uniformly spaced (eg. every hour, every day, etc.) such that the [`pandas.infer_freq`](https://pandas.pydata.org/docs/reference/api/pandas.infer_freq.html) function can be successfully applied to these time values. Missing target values should be in a format compatible with [pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html).
The ordering of the rows in your dataset does not matter (AutoGluon's forecasting models rely on the `time_column` value, not the row ordering).



### Can I use GPUs for model training?

Yes! Most of the models used by AutoGluon-Forecasting support GPU training, but it is not required that you train on a GPU. Make sure you have installed CUDA and the GPU version of MXNet, refer to the [installation instructions](../../install.html) for more details. AutoGluon will try to automatically detect whether your machine has a properly setup GPU, but you can also manually specify that GPU should be used via the `predictor.fit()` argument `ag_args_fit={'num_gpus': 1}`. Multi-GPU training is not yet supported.


### What machine is best for running AutoGluon Forecasting?

As an open-source library, AutoGluon-Forecasting can be run on any machine including your laptop. Currently it is not necessary to use a GPU to train forecasting models so CPU machines are fine albeit slower for certain models. We recommend running on a machine with as much memory as possible and the best available GPU (for instance if using AWS EC2, we recommend [P3 instances](https://aws.amazon.com/ec2/instance-types/p3/)).


### Issues not addressed here

First search if your issue is addressed in the [tutorials](index.html), [examples](https://github.com/awslabs/autogluon/tree/master/examples/forecasting), [documentation](../../api/autogluon.predictor.html), or [Github issues](https://github.com/awslabs/autogluon/issues) (search both Closed and Open issues). Also look through the [GluonTS Github issues](https://github.com/awslabs/gluon-ts/issues). If it is not there, please open a [new Github Issue](https://github.com/awslabs/autogluon/issues/new) and clearly state your issue and clarify it relates to forecasting. If you have a bug, please include: your code (ideally set `verbosity=4` which will print out more details), the output printed during the code execution, and information about your operating system, Python version, and installed packages (output of `pip freeze`). Many user issues stem from incorrectly formatted data, so please describe how your data look as clearly as possible.

pandas.Timestamp