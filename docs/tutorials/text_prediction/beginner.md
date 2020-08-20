# Text Prediction - Quick Start
:label:`sec_textprediction_beginner`

In this quick start, we will introduce the `TextPrediction` task in AutoGluon, which helps you solve NLP problems automatically.

The `TextPrediction` functionality depends on the [GluonNLP](https://gluon-nlp.mxnet.io/) package. 
Due to the ongoing upgrade of GluonNLP, we are currently using a customized version in [autogluon-contrib-nlp](https://github.com/sxjscience/autogluon-contrib-nlp.git). In a future release, we will switch to use the official GluonNLP.

In this example, we use two examples to show how to use `TextPrediction` to solve different types of NLP tasks, including:

- [Sentiment Analysis](#Sentiment-Analysis)
- [Sentence Similarity](#Sentence-Similarity)

The general usage is similar to AutoGluon-Tabular. We view NLP datasets as tables and specify that 
certain column is the label column.
Here, the label can not only be **categorical** but also **numerical**.
Internally, we build the network based on pretrained NLP models including [BERT](https://arxiv.org/pdf/1810.04805.pdf),
[ALBERT](https://arxiv.org/pdf/1909.11942.pdf), and [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB).
In addition, we run multiple trials with different hyper-parameters and return the best model. The searching logic is powered by the prebuilt Hyper-Parameter Optimization (HPO) algorithm.


```{.python .input}
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Sentiment Analysis

First, we use the Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)).
The dataset consists movie reviews and their sentiment. We will need to predict the correct sentiment.
It's a **binary classification** problem. Let's first load the data and view some examples.


```{.python .input}
from autogluon.utils.tabular.utils.loaders.load_pd import load
train_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
dev_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
rand_idx = np.random.permutation(np.arange(len(train_data)))[:2000]
train_data = train_data.iloc[rand_idx]
train_data.head(10)
```

For this example, we simply call `.fit()` with a random of 2000 training samples.
Each trial is trained for at most 1 minute using `.fit()`. 
Also, this tutorial is generated with a single [g4dn.xlarge](https://aws.amazon.com/ec2/instance-types/g4/) instance.
If you are just using machines with less computational resources, you may increase the training time to 1 hour
or do not specify `time_limits`.


```{.python .input}
from autogluon import TextPrediction as task

predictor = task.fit(train_data, label='label', time_limits='1min',
                     ngpus_per_trial=1, seed=123,
                     output_directory='./ag_sst')
```

Next, you can use `predictor.evaluate()` to evaluate the model on the dev set.


```{.python .input}
dev_score = predictor.evaluate(dev_data, metrics='acc')
print('Total Time = {}s'.format(predictor.results['total_time']))
print('Accuracy = {:.2f}%'.format(dev_score['acc'] * 100))
```

To get the prediction, use `predictor.predict()`.


```{.python .input}
sentence1 = "it's a charming and often affecting journey." 
sentence2 = "It's slow, very, very, very slow."
predictions = predictor.predict({'sentence': [sentence1, sentence2]})
print('"Sentence":', sentence1, '"Sentiment":', predictions[0])
print('"Sentence":', sentence2, '"Sentiment":', predictions[1])

```

For classification tasks, you can also choose to output the probability of each class via `predictor.predict_proba()`.


```{.python .input}
probs = predictor.predict_proba({'sentence': [sentence1, sentence2]})
print('"Sentence":', sentence1, '"Sentiment":', probs[0])
print('"Sentence":', sentence2, '"Sentiment":', probs[1])

```

## Sentence Similarity

Here, let's see how to use AutoGluon to train a model for evaluating the similarity between two sentences.
We use the [Semantic Textual Similarity Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) for illustration.


```{.python .input}
train_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet')[['sentence1', 'sentence2', 'score']]
dev_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet')[['sentence1', 'sentence2', 'score']]
train_data.head(10)
```

```{.python .input}
print('Min score=', min(train_data['score']), ', Max score=', max(train_data['score']))
```

Let's train a regression model with `task.fit()`. 
We will only need to specify the label column and AutoGluon will figure out the problem type and loss function automatically.

```{.python .input}
predictor_sts = task.fit(train_data, label='score',
                         time_limits=60, ngpus_per_trial=1, seed=123,
                         output_directory='./ag_sts')
```

Next, we call `.evalaute()` to calculate metrics on the dev dataset. We choose to report RMSE, Pearson's Correlation, and Spearman's Correlation.

```{.python .input}
dev_score = predictor_sts.evaluate(dev_data, metrics=['rmse', 'pearsonr', 'spearmanr'])
print('Best Config = {}'.format(predictor_sts.results['best_config']))
print('Total Time = {}s'.format(predictor_sts.results['total_time']))
print('RMSE = {:.2f}'.format(dev_score['rmse']))
print('PEARSONR = {:.4f}'.format(dev_score['pearsonr']))
print('SPEARMANR = {:.4f}'.format(dev_score['spearmanr']))
```

Next, we use our trained predictor to calculate the similarity score among these sentences:

- 'The child is riding a horse.'
- 'The young boy is riding a horse.'
- 'The young man is riding a horse.'
- 'The young man is riding a bicycle.'


```{.python .input}
sentences = ['The child is riding a horse.',
             'The young boy is riding a horse.',
             'The young man is riding a horse.',
             'The young man is riding a bicycle.']

score1 = predictor_sts.predict({'sentence1': [sentences[0]],
                                'sentence2': [sentences[1]]})

score2 = predictor_sts.predict({'sentence1': [sentences[0]],
                                'sentence2': [sentences[2]]})

score3 = predictor_sts.predict({'sentence1': [sentences[0]],
                                'sentence2': [sentences[3]]})
print(score1, score2, score3)
```

## Save and Load
Here's the basic usage of loading and saving the model.


```{.python .input}
predictor_sts.save('saved_dir')
predictor_sts_new = task.load('saved_dir')

score3 = predictor_sts_new.predict({'sentence1': [sentences[0]],
                                    'sentence2': [sentences[3]]})
print(score3)
```
