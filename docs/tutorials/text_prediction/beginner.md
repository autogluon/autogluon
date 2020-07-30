# Text Prediction - Quick Start
:label:`sec_textquick`

In this quick start, we will introduce the `TextPrediction` task in AutoGluon to illustrate basic
usage of AutoGluon’s NLP capability.

The AutoGluon Text functionality depends on the [GluonNLP](https://gluon-nlp.mxnet.io/) package. 
We are currently using a customized version in [autogluon-contrib-nlp](https://github.com/sxjscience/autogluon-contrib-nlp.git).
In a future release, we will switch to use the official GluonNLP.

In this example, we use three examples to show how to use `TextPrediction` to solve different types
of NLP tasks, including:
 
- Sentiment Analysis
- Sentence Similarity
- Paraphrasing Detection

We  load sentences and the corresponding labels (sentiment) into AutoGluon and 
use this data to obtain a neural network that can classify new sentences. 
Different from traditional machine learning where we need to manually define the neural network,
and specify the hyperparameters in the training process, with just a single call to
`AutoGluon`'s `fit` function, AutoGluon will automatically give you a model that performs the best.

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

# For this simple example, we train for 5 minutes.
predictor = task.fit(train_data, label='label', time_limits = 5 * 60)
dev_score = predictor.evaluate(dev_data)
```

## Customization the Search Space
