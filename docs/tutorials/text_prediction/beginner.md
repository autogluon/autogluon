# Text Prediction - Quick Start
:label:`sec_textquick`

In this quick start, we will introduce the `TextPrediction` task in AutoGluon to illustrate basic usage of AutoGluon’s NLP capability.

The AutoGluon Text functionality depends on the [GluonNLP](https://gluon-nlp.mxnet.io/) package. 

In this tutorial, we are using sentiment analysis as a text classification example. We will load sentences and the 
corresponding labels (sentiment) into AutoGluon and use this data to obtain a neural network that can classify new sentences. 
Different from traditional machine learning where we need to manually define the neural network, and specify 
the hyperparameters in the training process, with just a single call to `AutoGluon`'s `fit` function, 
AutoGluon will automatically train many models under thousands of different hyperparameter configurations and then return the best model.

We begin by specifying `TextPrediction` as our task of interest:

```{.python .input}
import autogluon as ag
from autogluon import TextPrediction as task
```


## Use AutoGluon to fit Models
We use the Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)).
The original dataset consists of sentences from movie reviews and human annotations of their sentiment.
The task is to classify whether a given sentence has positive or negative sentiment (binary classification).

```{.python .input}
train_data = 'https://autogluon-text.s3-us-west-2.amazonaws.com/glue/sst/train.parquet'
dev_data = 'https://autogluon-text.s3-us-west-2.amazonaws.com/glue/sst/dev.parquet'
test_data = 'https://autogluon-text.s3-us-west-2.amazonaws.com/glue/sst/test.parquet'

predictor = task.fit(train_data, label='label')
predictions = predictor.predict(dev_data)
```

## Customization the Search Space
