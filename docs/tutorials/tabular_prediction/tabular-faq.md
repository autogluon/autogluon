# FAQ
:label:`sec_faq`


### How can I get the most accurate predictions?

See ["Maximizing predictive performance" in the Quick Start Tutorial](tabular-quickstart.html#maximizing-predictive-performance).


### Can I run AutoGluon Tabular on Mac/Windows?

Yes! The only functionality that may not work is hyperparameter tuning with the NN model (this should be resolved in the next MXNet update).


### What machine is best for running AutoGluon Tabular?

As an open-source library, AutoGluon can be run on any machine including your laptop. Currently the Tabular module does not benefit much from GPUs, so CPU machines are fine (in contrast, TextPrediction/ImagePrediction/ObjectDetection do greatly benefit from GPUs). Most Tabular issues arise due to lack of memory, so we recommend running on a machine with as much memory as possible. For example if using AWS instances for Tabular: we recommend [M5 instances](https://aws.amazon.com/ec2/instance-types/m5/), where a **m5.24xlarge** machine should be able to handle most datasets.


### How to resolve memory issues?

See ["If you encounter memory issues" in the In Depth Tutorial](tabular-indepth.html#if-you-encounter-memory-issues).


### How to resolve disk space issues?

See ["If you encounter disk space issues" in the In Depth Tutorial](tabular-indepth.html#if-you-encounter-disk-space-issues).


### How can I reduce the time required for training?

Specify the `time_limit` argument in `fit()` to the number of seconds you are willing to wait (longer time limits generally result in superior predictive performance). You may also try other settings of the `presets` argument in `fit()`, and can also subsample your data for a quick trial run via `train_data.sample(n=SUBSAMPLE_SIZE)`. If a particular type of model is taking much longer to train on your data than the other types of models, you can tell AutoGluon not to train any models of this particular type by specifying its short-name in the `excluded_model_types` argument of `fit()`.

Since many of the strategies to reduce memory usage also reduce training times, also check out: ["If you encounter memory issues" in the In Depth Tutorial](tabular-indepth.html#if-you-encounter-memory-issues).


### How can I reduce the time required for prediction?

See ["Accelerating inference" in the In Depth Tutorial](tabular-indepth.html#accelerating-inference).


### How does AutoGluon Tabular work internally?

Details are provided in the following paper:

[AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/abs/2003.06505). *Arxiv*, 2020.


### How to view more detailed logs of what is happening during fit?

Specify the argument `verbosity = 4` in `fit()`.


### What model is AutoGluon using for prediction?

See ["Prediction options" in the In Depth Tutorial](tabular-indepth.html#prediction-options-inference).


### Which classes do predicted probabilities correspond to?

This should become obvious if you specify the `as_pandas` argument like this:

```
predictor.predict_proba(test_data, as_pandas=True)
```

For multiclass classification:

```
predictor.class_labels
```

is a list of classes whose order corresponds to columns of `predict_proba()` output when it is a Numpy array. For binary classification, this is the order of classes when `predict_proba(as_multiclass=True)` is called.

You can see which class AutoGluon treats as the positive class in binary classification via:

```
predictor.positive_class
```

The positive class can also be retrieved via `predictor.class_labels[-1]`. The output of `predict_proba()` for binary classification is the probability of the positive class.

### How can I use AutoGluon for interpretability?

See ["Interpretability (feature importance)" in the In Depth Tutorial](tabular-indepth.html#interpretability-feature-importance), which allows you to quantify how much each feature contributes to AutoGluon's predictive accuracy.

Additionally, you can explain particular AutoGluon predictions using [Shapely values](https://github.com/slundberg/shap/). Notebooks demonstrating this are provided at: [https://github.com/awslabs/autogluon/tree/master/examples/tabular/interpret](https://github.com/awslabs/autogluon/tree/master/examples/tabular/interpret). Handling of multiclass classification tasks and data with categorical features are demonstrated in the notebook "SHAP with AutoGluon-Tabular and Categorical Features" contained in this folder.




### How can I perform inference on a file that won't fit in memory?

The Tabular Dataset API works with pandas Dataframes, which supports chunking data into sizes that fit in memory.
Here's an example of one such chunk-based inference:

```{.python .input}
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import requests

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
predictor = TabularPredictor(label='class').fit(train_data.sample(n=100, random_state=0), hyperparameters={'GBM': {}})

# Get the test dataset, if you are working with local data then omit the next two lines
r = requests.get('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv', allow_redirects=True)
open('test.csv', 'wb').write(r.content)
reader = pd.read_csv('test.csv', chunksize=1024)
y_pred = []
y_true = []
for df_chunk in reader:
    y_pred.append(predictor.predict(df_chunk))
    y_true.append(df_chunk['class'])
y_pred = pd.concat(y_pred, axis=0, ignore_index=True)
y_true = pd.concat(y_true, axis=0, ignore_index=True)
predictor.evaluate_predictions(y_true=y_true, y_pred=y_pred)
```

Here we split the test data into chunks of up to 1024 rows each, but you may select a larger size as long as it fits into your system's memory.
[Further Reading](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-chunking)


### How can I skip some particular models?

To avoid training certain models, specify these in the `excluded_model_types` argument. For example, here's how to call `fit()` without training K Nearest Neighbor (KNN), Random Forest (RF), or ExtraTrees (XT) models:

```
task.fit(..., excluded_model_types=['KNN','RF','XT'])
```

### How can I add my own custom model to the set of models that AutoGluon trains, tunes, and ensembles?

See this example in the source code: [examples/tabular/example_custom_model_tabular.py](https://github.com/awslabs/autogluon/blob/master/examples/tabular/example_custom_model_tabular.py)


### How can I add my own custom data preprocessing or feature engineering?

Note that the `TabularDataset` object is essentially a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html) and you can transform your training data however you wish before calling `fit()`. Note that any transformations you perform yourself must also be applied to all future test data before calling `predict()`, and AutoGluon will still perform its default processing on your transformed data inside `fit()`.

To solely use custom data preprocessing and automatically apply your custom transformations to both the train data and all future data encountered during inference, you should instead create a custom FeatureGenerator. Follow this example in the source code: [examples/tabular/example_custom_feature_generator.py](https://github.com/awslabs/autogluon/blob/master/examples/tabular/example_custom_feature_generator.py)

### I'm receiving C++ warning spam during training or inference

Warning message: [W ParallelNative.cpp:206] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)

This can happen from downstream PyTorch dependencies (OpenMP) when using a specific environment. If you are using PyTorch 1.7, Mac OS X, Python 3.6/3.7, and using the PyTorch DataLoader, then you may get this warning spam. We have only seen this occur with the TabTransformer model. Reference open [torch issue](https://github.com/pytorch/pytorch/issues/46409).

The recommended workaround from the torch issue to suppress this warning is to set an environment variable used in OpenMP. This has been tested on Torch 1.7, Python 3.6, and Mac OS X.
```
export OMP_NUM_THREADS=1
```

### Issues not addressed here

First search to see if your issue is addressed in the other [tutorials](index.html)/[documentation](../../api/autogluon.task.html), or the [Github issues](https://github.com/awslabs/autogluon/issues). If it is not there,
please open a [new Github Issue](https://github.com/awslabs/autogluon/issues/new) and clearly state your issue. If you have a bug, please include: your code (call `fit(..., verbosity=4)` which will print more details), the output printed during the code execution, and information about your operating system, Python version, and installed packages (output of `pip freeze`).
