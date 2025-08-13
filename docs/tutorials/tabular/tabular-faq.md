# AutoGluon Tabular FAQ


## How can I get the most accurate predictions?

See ["Maximizing predictive performance" in the Tabular Essentials tutorial](tabular-essentials.ipynb).


## Can I run AutoGluon Tabular on Mac/Windows?

Yes!

## Can I use GPUs for model training?

Yes! Most of the models used by AutoGluon support GPU training, including LightGBM, CatBoost, XGBoost and FastAI Neural Network.

To enable GPU training, specify in [predictor.fit](../../api/autogluon.tabular.TabularPredictor.fit.rst) the argument `num_gpus=SOME_VALUE`. This will enable GPU training for all models that support it. Multi-GPU training is still experimental.

For most of these models, CUDA will have to be installed and some models may need special installations such as LightGBM to be compatible with GPU training. Refer to [installation instructions](../../install.md) for more details.


## What machine is best for running AutoGluon Tabular?

As an open-source library, AutoGluon can be run on any machine including your laptop. Currently it is not necessary to use a GPU to train TabularPredictor so CPU machines are fine (in contrast, MultiModalPredictor requires GPUs). Most Tabular issues arise due to lack of memory, so we recommend running on a machine with as much memory as possible. For example if using AWS instances for Tabular: we recommend [M5 instances](https://aws.amazon.com/ec2/instance-types/m5/), where a **m5.24xlarge** machine should be able to handle most datasets.


## How can I resolve memory issues?

See ["If you encounter memory issues" in the In Depth Tutorial](tabular-indepth.ipynb).


## How can I resolve disk space issues?

See ["If you encounter disk space issues" in the In Depth Tutorial](tabular-indepth.ipynb).


## How can I reduce the time required for training?

Specify the `time_limit` argument in [predictor.fit](../../api/autogluon.tabular.TabularPredictor.fit.rst) to the number of seconds you are willing to wait (longer time limits generally result in superior predictive performance). You may also try other settings of the `presets` argument in [predictor.fit](../../api/autogluon.tabular.TabularPredictor.fit.rst), and can also subsample your data for a quick trial run via `train_data.sample(n=SUBSAMPLE_SIZE)`. If a particular type of model is taking much longer to train on your data than the other types of models, you can tell AutoGluon not to train any models of this particular type by specifying its short-name in the `excluded_model_types` argument of `fit()`.

Since many of the strategies to reduce memory usage also reduce training times, also check out: ["If you encounter memory issues" in the In Depth Tutorial](tabular-indepth.ipynb).


## How can I reduce the time required for prediction?

See ["Accelerating inference" in the In Depth Tutorial](tabular-indepth.ipynb).

## How can I use the foundational models?

See [Foundational Models Tutorial](tabular-foundational-models.ipynb).


## How does AutoGluon Tabular work internally?

Details are provided in the following paper:

[AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/abs/2003.06505). *Arxiv*, 2020.


## How can I view more detailed logs of what is happening during fit?

Specify the argument `verbosity = 4` in `fit()`.


## What model is AutoGluon using for prediction?

See ["Prediction options" in the In Depth Tutorial](tabular-indepth.ipynb).


## Which classes do predicted probabilities correspond to?

This should become obvious if you look at the pandas DataFrame column names from the prediction probability output:

```
predictor.predict_proba(test_data)
```

For binary and multiclass classification:

```
predictor.class_labels
```

is a list of classes whose order corresponds to columns of `predict_proba(as_pandas=False)` output when it is a Numpy array.

You can see which class AutoGluon treats as the positive class in binary classification via:

```
predictor.positive_class
```

The positive class can also be retrieved via `predictor.class_labels[-1]`. The output of `predict_proba(as_multiclass=False)` for binary classification is the probability of the positive class.


## How can I use AutoGluon for interpretability?

See ["Interpretability (feature importance)" in the In Depth Tutorial](tabular-indepth.ipynb), which allows you to quantify how much each feature contributes to AutoGluon's predictive accuracy.

Additionally, you can explain particular AutoGluon predictions using [Shapely values](https://github.com/slundberg/shap/). Notebooks demonstrating this are provided at: [https://github.com/autogluon/autogluon/tree/master/examples/tabular/interpret](https://github.com/autogluon/autogluon/tree/master/examples/tabular/interpret). We recommend starting with the notebook "SHAP with AutoGluon-Tabular" contained in this folder, which demonstrates handling of multiclass classification tasks and data with categorical features.


## How can I perform inference on a file that won't fit in memory?

The Tabular Dataset API works with pandas DataFrames, which supports chunking data into sizes that fit in memory.
Here's an example of one such chunk-based inference:

```python
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


## How can I skip some particular models?

To avoid training certain models, specify these in the `excluded_model_types` argument. For example, here's how to call `fit()` without training K Nearest Neighbor (KNN), Random Forest (RF), or ExtraTrees (XT) models:

```
task.fit(..., excluded_model_types=['KNN','RF','XT'])
```

## How can I add my own custom model to the set of models that AutoGluon trains, tunes, and ensembles?

To learn how to add your own custom models to the set that AutoGluon trains, tunes, and ensembles, review [Tabular Custom Model](advanced/tabular-custom-model).

## How can I reproduce the results of model XYZ without using AutoGluon?

Exactly reproducing the results of AutoGluon models without using AutoGluon can be challenging
(Ex: Trying to replicate the results of AutoGluon's LightGBM without depending on AutoGluon).
AutoGluon uses a variety of techniques not present in the model frameworks it builds upon such as data preprocessing,
edge case handling, custom early stopping, and custom architectures. These are all used to enhance the performance
and simplify usage beyond what exists in the original model framework (such as LightGBM, XGBoost, FastAI, etc.).
This is a core benefit of AutoML frameworks and is inherently complicated to replicate
(you would have to reimplement much of what AutoGluon does behind the scenes).
If you still wish to try to replicate results, the logic can be derived from the source code.

## Why was XYZ design decision chosen?

Usually the answer boils down to
"because it performed better on average than the alternatives we tested across a wide variety of benchmark datasets"
and/or "because it is simple and easy to implement while likely performing similarly to more complex methods"
and/or "because it worked well enough and we haven't investigated further options".
AutoML is inherently an empirically oriented field rather than a theoretical one.
This is because theoretically explaining why the various AutoML components work together better than numerous
alternatives in practice is not particularly feasible nor generalizable.

## How can I add my own custom data preprocessing or feature engineering?

Note that the `TabularDataset` object is essentially a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html) and you can transform your training data however you wish before calling `fit()`. Note that any transformations you perform yourself must also be applied to all future test data before calling `predict()`, and AutoGluon will still perform its default processing on your transformed data inside `fit()`.

To solely use custom data preprocessing and automatically apply your custom transformations to both the train data and all future data encountered during inference, you should instead create a custom FeatureGenerator. Follow this example in the source code: [examples/tabular/example_custom_feature_generator.py](https://github.com/autogluon/autogluon/blob/master/examples/tabular/example_custom_feature_generator.py)

## How can I differently weight the importance of training examples?

You can specify the `sample_weight` and `weight_evaluation` [arguments](../../api/autogluon.tabular.TabularPredictor.rst) when initializing a `TabularPredictor`.

## How does missing value imputation work in AutoGluon?

AutoGluon does not perform generic missing value imputation, instead it sends the missing values to each model,
and each model has different custom handling of missing values.
This improves model diversity and thus the final strength of the ensemble.
Refer to the model source code to learn how missing values are handled.

## How to limit the number of cores AutoGluon will use
Although it is generally recommended to let AutoGluon to use all the cores, you can limit it by setting the `num_cpus`:
```
predictor = TabularPredictor(...).fit(..., num_cpus = NUM_CORES_YOU_WANT)
```
You can also limit the number of cores used by a specific model:
```
# We use 1 core for CatBoost model, 4 cores for XGBoost model, and all cores for lightGBM model here.
predictor = TabularPredictor(...).fit(..., hyperparameters= {'CAT': {'ag_args_fit': {'num_cpus': 1}}, 'XGB': {'ag_args_fit': {'num_cpus': 4}}, 'GBM': {}},)
```

## My custom metric is raising `_pickle.PicklingError: Can't pickle` errors
See ["Ensuring Metric is Serializable" in Custom Metric Tutorial](advanced/tabular-custom-metric.ipynb).

## I know AutoGluon uses ray underneath. What's the best practice for me?
It is generally recommended to not have your custom usage of ray resources, i.e. init ray cluster, along with AutoGluon in a same script.
Ray does not support multiple runtimes in a same script, hence unexpected behaviors could happen.

## No space left error on SageMaker Notebook
If you are using AutoGluon on SageMaker Notebook, it is likely that you will encounter such error: `OSError: [Errno 28] No space left on device`. This is because the default disk size of a SageMaker Notebook instance is 5 GiB regardless of the type. AutoGluon training on some large datasets could end up with artifacts that's larger than 5GiB.

To address it, either cleanup your workspace, or 1) shutdown your Notebook instance 2) choose your Notebook instance 3) update the `Volume size in GB` field under `Edit`

## Issues not addressed here

First search if your issue is addressed in the [tutorials](index.md), [examples](https://github.com/autogluon/autogluon/tree/master/examples/tabular), [documentation](../../api/autogluon.tabular.TabularPredictor.rst), or [Github issues](https://github.com/autogluon/autogluon/issues) (search both Closed and Open issues). If it is not there, please open a [new Github Issue](https://github.com/autogluon/autogluon/issues/new) and clearly state your issue. If you have a bug, please include: your code (call `fit(..., verbosity=4)` which will print more details), the output printed during the code execution, and information about your operating system, Python version, and installed packages (output of `pip freeze`).
