# Text Classification - Quick Start
:label:`sec_textquick`

Note: `TextClassification` is in preview mode and is not feature complete. While the tutorial described below is functional, using `TextClassification` on custom datasets is not yet supported.
For an alternative, text data can be passed to `TabularPrediction` in tabular format which has text feature support.

We adopt the task of Text Classification as a running example to illustrate basic usage of AutoGluon’s NLP capability.

The AutoGluon Text functionality depends on the [GluonNLP](https://gluon-nlp.mxnet.io/) package. 
Thus, in order to use AutoGluon-Text, you will need to install GluonNLP via `pip install gluonnlp==0.8.1`

In this tutorial, we are using sentiment analysis as a text classification example. We will load sentences and the 
corresponding labels (sentiment) into AutoGluon and use this data to obtain a neural network that can classify new sentences. 
Different from traditional machine learning where we need to manually define the neural network, and specify 
the hyperparameters in the training process, with just a single call to `AutoGluon`'s `fit` function, 
AutoGluon will automatically train many models under thousands of different hyperparameter configurations and then return the best model.

We begin by specifying `TextClassification` as our task of interest:

```{.python .input}
import autogluon as ag
from autogluon import TextClassification as task
```


## Create AutoGluon Dataset
We are using a subset of the Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)).
The original dataset consists of sentences from movie reviews and human annotations of their sentiment.
The task is to classify whether a given sentence has positive or negative sentiment (binary classification).

```{.python .input}
dataset = task.Dataset(name='ToySST')
```

In the above call, we have the proper train/validation/test split of the SST dataset.


## Use AutoGluon to fit Models

Now, we want to obtain a neural network classifier using AutoGluon. In the default configuration, rather than attempting to train complex models from scratch using our data, AutoGluon fine-tunes neural networks that have already been pretrained on large scale text dataset such as Wikicorpus. Although the dataset involves entirely different text, lower-level features captured in the representations of the pretrained network (such as edge/texture detectors) are likely to remain useful for our own text dataset.  

While we primarily stick with default configurations in this Beginner tutorial, the Advanced tutorial covers various options that you can specify for greater control over the training process. With just a single call to AutoGluon's `fit` function, AutoGluon will train many models with different hyperparameter configurations and return the best model.

However, neural network training can be quite time-costly. To ensure quick runtimes,
we tell AutoGluon to obey strict limits: `epochs` specifies how much
computational effort can be devoted to training any single network, while `time_limits`
in seconds specifies how much time `fit` has to return a model (more
precisely, training runs are started as long as `time_limits` is not reached).
For demo purposes, we specify only small values for `time_limits`, `epochs`:

```{.python .input}
predictor = task.fit(dataset, epochs=1, time_limits=30)
```

Within `fit`, the model with the best hyperparameter configuration is selected based on its validation accuracy after being trained on the data in the training split.  

The best Top-1 accuracy achieved on the validation set is:

```{.python .input}
print('Top-1 val acc: %.3f' % predictor.results['best_reward'])
```

Within `fit`, this model is also finally fitted on our entire dataset (i.e., merging training+validation) using the same optimal hyperparameter configuration. The resulting model is considered as final model to be applied to classify new text.

We now construct a test dataset similarly as we did with the train dataset, and then `evaluate` the final model produced by `fit` on the test data:

```{.python .input}
test_acc = predictor.evaluate(dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

Given an example sentence, we can easily use the final model to `predict` the label (and the conditional class-probability):

```{.python .input}
sentence = 'I feel this is awesome!'
ind = predictor.predict(sentence)
print('The input sentence sentiment is classified as [%d].' % ind.asscalar())
```

The `results` object returned by `fit` contains summaries describing various aspects of the training process.
For example, we can inspect the best hyperparameter configuration corresponding to the final model which achieved the above (best) results:

```{.python .input}
print('The best configuration is:')
print(predictor.results['best_config'])
```

