# Text Classification - Quick Start
:label:`sec_textquick`

```python
from autogluon import text_classification as task

import logging
logging.basicConfig(level=logging.INFO)
```


## Create AutoGluon Dataset

```python
dataset = task.Dataset(name='sst')
```

In the above call, we have the proper train/validation/test split of the sst dataset.


## Use AutoGluon to fit models

Now, we want to obtain a neural network classifier using AutoGluon. In the default configuration, rather than attempting to train complex models from scratch on our data, AutoGluon will instead fine-tune neural networks that have already been pretrained on large scale text dataset such as Wikicorpus. Although the dataset involves entirely different text, the idea here is that lower-level features captured in the representations of the pretrained network (such as edge/texture detectors) are likely to remain useful for our own text dataset.  

While we stick with mostly default configurations in this Beginner tutorial, the Advanced tutorial will cover various options that you can specify for greater control over the training process. With just a single call to AutoGluon's `fit` function, AutoGluon will train many models with different hyperparameter configurations and return the best model.

However, neural network training can be quite time-costly. To ensure quick runtimes, we tell AutoGluon to obey strict limits: `num_training_epochs` specifies how much computational effort can be devoted to training any single network, while `time_limits` in seconds specifies how much time `fit` has to return a model. For demo purposes, we specify only small values for `time_limits`, `num_training_epochs`:

```python
time_limits = 3*60 # 3mins
num_training_epochs = 10
results = task.fit(dataset,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)
```

Within `fit`, the model with the best hyperparameter configuration is selected based on its validation accuracy after being trained on the data in the training split.  

The best Top-1 accuracy achieved on the validation set is:

```python
print('Top-1 val acc: %.3f' % results.metric)
```

Within `fit`, this model is also finally fitted on our entire dataset (ie. merging training+validation) using the same optimal hyperparameter configuration. The resulting model is considered as final model to be applied to classify new text.

We now construct a test dataset similarly as we did with the train dataset, and then `evaluate` the final model produced by `fit` on the test data:

```python
test_acc = task.evaluate(dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

Given an example sentence, we can easily use the final model to `predict` the label (and the conditional class-probability):

```python
sentence = 'I feel this is awesome!'
ind, prob = task.predict(sentence)
print('The input sentence is classified as [%s], with probability %.2f.' %
      (dataset.train.synsets[ind.asscalar()], prob.asscalar()))
```

The `results` object returned by `fit` contains summaries describing various aspects of the training process.
For example, we can inspect the best hyperparameter configuration corresponding to the final model which achieved the above results:

```python
print('The best configuration is:')
print(results.config)
```

This configuration is used to generate the above results.