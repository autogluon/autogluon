# FAQ
:label:`sec_faq`


### How can I get the most accurate predictions?

See ["Maximizing predictive performance" in the Quick Start Tutorial](tabular-quickstart.html#maximizing-predictive-performance).


### Can I run TabularPrediction on Mac/Windows?

Yes! The only functionality that may not work is `hyperparameter_tune=True` with the NN model (this should be resolved in the next MXNet update).


### What machine is best for running TabularPrediction?

As an open-source library, AutoGluon can be run on any machine. Currently the TabularPrediction module does not benefit much from GPUs, so CPU machines are fine (in contrast, TextPrediction/ImageClassification/ObjectDetection all do greatly benefit from GPUs). Most issues arise due to lack of memory, so we recommend running on a machine with as much memory as possible. For example if using AWS instances for TabularPrediction: we recommend [M5 instances](https://aws.amazon.com/ec2/instance-types/m5/), where a **m5.24xlarge** machine should be able to handle most datasets.


### How can I reduce the time required for prediction?

See ["Accelerating inference" in the In Depth Tutorial](tabular-indepth.html#accelerating-inference).


### How to resolve memory issues?

See ["If you encounter memory issues" in the In Depth Tutorial](tabular-indepth.html#if-you-encounter-memory-issues).


### How to resolve disk space issues?

See ["If you encounter disk space issues" in the In Depth Tutorial](tabular-indepth.html#if-you-encounter-disk-space-issues).


### How does TabularPrediction work internally?

Details are provided in the following paper:

[AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/abs/2003.06505). *Arxiv*, 2020.


### What model is AutoGluon using for prediction?

See ["Prediction options" in the In Depth Tutorial](tabular-indepth.html#prediction-options-inference).


### Which classes do predicted probabilities correspond to?

This should become obvious if you ask for predictions like this:

```
predictor.predict_proba(test_data, as_pandas = True)
```

Alternatively, you can see which class AutoGluon treats as the positive class in binary classification via:

```
positive_class = [label for label in predictor.class_labels if predictor.class_labels_internal_map[label]==1][0]
```

Or for multiclass classification:
```
predictor.class_labels
```
is a list of classes whose order corresponds to columns of `predict_proba()` output when it is a Numpy array.


### How can I use AutoGluon for interpretability?

See ["Interpretability (feature importance)" in the In Depth Tutorial](tabular-indepth.html#interpretability-feature-importance).


### How can I perform inference on a file that won't fit in memory?

The Tabular Dataset API works with pandas Dataframes, which supports chunking data into sizes that fit in memory.
Here's an example of one such chunk-based inference:

```{.python .input}
from autogluon import TabularPrediction as task
import pandas as pd
import requests

train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
predictor = task.fit(train_data=train_data.sample(n=100, random_state=0), label='class', hyperparameters={'GBM': {}})

# Get the test dataset, if you are working with local data then omit the next two lines
r = requests.get('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv', allow_redirects=True)
open('test.csv', 'wb').write(r.content)
reader = pd.read_csv('test.csv', chunksize=1024)
y_pred = []
y_true = []
for df_chunk in reader:
    y_pred.append(predictor.predict(df_chunk, as_pandas=True))
    y_true.append(df_chunk['class'])
y_pred = pd.concat(y_pred, axis=0, ignore_index=True)
y_true = pd.concat(y_true, axis=0, ignore_index=True)
predictor.evaluate_predictions(y_true=y_true, y_pred=y_pred)
```

Here we split the test data into chunks of up to 1024 rows each, but you may select a larger size as long as it fits into your system's memory.
[Further Reading](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-chunking)
