# FAQ - Time Series
:label:`sec_forecastingfaq`


### Where can I find more information about the models/metrics?

Metrics are implemented in the `autogluon.timeseries.evaluator` module. We also follow some of
the same conventions followed by GluonTS in their evaluation.
Please refer to
the GluonTS [documentation](https://ts.gluon.ai/stable/api/gluonts/gluonts.html) and
[github](https://github.com/awslabs/gluon-ts) for further information.

A detailed description of evaluation metrics is also available at
[here](https://docs.aws.amazon.com/forecast/latest/dg/metrics.html).

### How can I get the most accurate forecast predictions?

Generally setting the `predictor.fit()` argument `presets="best_quality"` will result in high accuracy.
Alternative options include manually specifying hyperparameter search spaces for certain models and
manually increasing the number of hyperparameter optimization trials.


### Can I use GPUs for model training?

Yes! Most of the models used by AutoGluon-Forecasting support GPU training, but it is not
required that you train on a GPU. Make sure you have installed CUDA and the GPU version of MXNet.
AutoGluon will try to automatically detect whether your machine has a GPU, and train
neural network based models on these. Multi-GPU training is not yet supported.


### What machine is best for running AutoGluon-Forecasting?

As an open-source library, AutoGluon-Forecasting can be run on any machine including your laptop.
Currently it is not necessary to use a GPU to train forecasting models so CPU machines are fine
albeit slower for certain models. We recommend running on a machine with as much memory as possible
and the best available GPU (for instance if using AWS EC2, we
recommend [P3 instances](https://aws.amazon.com/ec2/instance-types/p3/)).


### Issues not addressed here

First search if your issue is addressed in the [tutorials](index.html),
[examples](https://github.com/awslabs/autogluon/tree/master/examples/forecasting),
[documentation](../../api/autogluon.predictor.html), or [Github issues](https://github.com/awslabs/autogluon/issues)
(search both Closed and Open issues).
If it is not there, please open a [new Github Issue](https://github.com/awslabs/autogluon/issues/new) and
clearly state your issue and clarify it relates to forecasting.

If you have a bug, please include: your code (ideally set `verbosity=4` which will print out more details), the
output printed during the code execution, and information about your operating system, Python version, and
installed packages (output of `pip freeze`).
Many user issues stem from incorrectly formatted data, so please describe your data as clearly as possible.
