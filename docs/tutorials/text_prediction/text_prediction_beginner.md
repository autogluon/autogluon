# Text Prediction - Quick Start
:label:`sec_textprediction_quick`

In this quick start, we will introduce the `TextPrediction` task in AutoGluon to illustrate basic usage of AutoGluon’s NLP capability.

The `TextPrediction` functionality depends on the [GluonNLP](https://gluon-nlp.mxnet.io/) package. 
Due to the on-going upgrade of GluonNLP, we are currently using a customized version in [autogluon-contrib-nlp](https://github.com/sxjscience/autogluon-contrib-nlp.git). In a future release, we will switch to use the official GluonNLP.

In this example, we use three examples to show how to use `TextPrediction` to solve different types of NLP tasks, including:
- Sentiment Analysis
- Paraphrasing Identification
- Sentence Similarity

The general usage is similar to AutoGluon-Tabular, we load the datasets as a pandans table and specify that certain column is the label column. Here, the label can not only be categorical but also numerical. Internally, we build the network based on pretrained NLP models like [BERT](https://arxiv.org/pdf/1810.04805.pdf), [ALBERT](https://arxiv.org/pdf/1909.11942.pdf), [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB), and applies Hyper-parameter Optimization (HPO) to search for the best configuration.


```python
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)

import autogluon as ag
from autogluon import TextPrediction as task
```

## Sentiment Analysis

First, we use the Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)).
The dataset consists movie reviews and their sentiment. We will need to predict the correct sentiment. It's a **binary classification** problem. Let's first load the data and view some examples.


```python
from autogluon.utils.tabular.utils.loaders import load_pd

train_data = load_pd.load('https://autogluon-text.s3-us-west-2.amazonaws.com/glue/sst/train.parquet')
dev_data = load_pd.load('https://autogluon-text.s3-us-west-2.amazonaws.com/glue/sst/dev.parquet')
train_data.head(10)
```

For this example, we simply call `.fit()` with a random of 1000 training samples. Each trial is trained for 1 minute using `.fit()` and we will just use 3 trials.


```python
rand_idx = np.random.permutation(np.arange(len(train_data)))[:1000]
train_data = train_data.iloc[rand_idx]

predictor = task.fit(train_data, label='label', num_trials=3, time_limits=60)
```

Next, you may use `predictor.evaluate()` to evaluate the model on the dev set.


```python
dev_score = predictor.evaluate(dev_data, metrics='acc')
print(dev_score)
```


```python
predictions = predictor.predict(dev_data)
print('"Sentence":', dev_data['sentence'].iloc[0], '"Sentiment":', predictions[0])
```

## Paraphrasing Identification

The Paraphrasing Identification task is to identify whether two 


```python
train_data = load_pd.load('https://autogluon-text.s3-us-west-2.amazonaws.com/glue/mrpc/train.parquet')
dev_data = load_pd.load('https://autogluon-text.s3-us-west-2.amazonaws.com/glue/mrpc/dev.parquet')

```
