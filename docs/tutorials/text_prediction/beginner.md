# Text Prediction - Quick Start
:label:`sec_textprediction_beginner`

Here we introduce the `TextPrediction` task, which helps you automatically train and deploy models for various Natural Language Processing (NLP) problems.
This tutorial presents two examples to demonstrate how `TextPrediction` can be used for different NLP tasks including:

- [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
- [Sentence Similarity](https://arxiv.org/abs/1910.03940)

The general usage is similar to AutoGluon's `TabularPredictor`. We treat NLP datasets as tables where certain columns contain text fields and a special column contains the labels to predict. 
Here, the labels can be discrete categories (classification) or numerical values (regression).
`TextPrediction` fits neural networks to your data via transfer learning from pretrained NLP models like: [BERT](https://arxiv.org/pdf/1810.04805.pdf),
[ALBERT](https://arxiv.org/pdf/1909.11942.pdf), and [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB).
`TextPrediction` also trains multiple models with different hyperparameters and returns the best model, a process called Hyperparameter Optimization (HPO).


```{.python .input}
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Sentiment Analysis

First, we consider the Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)) dataset, which consists of movie reviews and their associated sentiment. Given a new movie review, the goal is to predict the sentiment reflected in the text (in this case a **binary classification** problem, where reviews are labeled as 1 if they convey a positive opinion and labeled as 0 otherwise). Let's first load the data and view some examples, noting the labels are stored in a column called **label**.


```{.python .input}
from autogluon.core.utils.loaders.load_pd import load
train_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
dev_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
rand_idx = np.random.permutation(np.arange(len(train_data)))[:1000]
train_data = train_data.iloc[rand_idx]
train_data.head(10)
```

Above the data happen to be stored in a [Parquet](https://databricks.com/glossary/what-is-parquet) table format, but you can also directly `load()` data from a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file instead. While here we load files from [AWS S3 cloud storage](https://docs.aws.amazon.com/AmazonS3/latest/dev/Welcome.html), these could instead be local files on your machine. After loading, `train_data` is simply a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), where each row represents a different training example (for machine learning to be appropriate, the rows should be independent and identically distributed).

To ensure this tutorial runs quickly, we simply call `fit()` with a subset of 2000 training examples and limit its runtime to approximately 1 minute. 
To achieve reasonable performance in your applications, you should set much longer `time_limits` (eg. 1 hour), or do not specify `time_limits` at all.


```{.python .input}
from autogluon.text import TextPrediction as task

predictor = task.fit(train_data, label='label', 
                     time_limits=60,
                     ngpus_per_trial=1,
                     seed=123,
                     output_directory='./ag_sst')
```

Above we specify that: the **label** column of our DataFrame contains the label-values to predict, AutoGluon should run for 60 seconds, each training run of an individual model (with particular hyperparameters) should run on 1 GPU, a particular random seed should be used to facilitate reproducibility, and that trained models should be saved in the **ag_sst** folder.

Now you can use `predictor.evaluate()` to evaluate the trained model on some separate test data.


```{.python .input}
dev_score = predictor.evaluate(dev_data, metrics='acc')
print('Total Time = {}s'.format(predictor.results['total_time']))
print('Accuracy = {:.2f}%'.format(dev_score['acc'] * 100))
```

And you can easily obtain predictions from these models.


```{.python .input}
sentence1 = "it's a charming and often affecting journey." 
sentence2 = "It's slow, very, very, very slow."
predictions = predictor.predict({'sentence': [sentence1, sentence2]})
print('"Sentence":', sentence1, '"Predicted Sentiment":', predictions[0])
print('"Sentence":', sentence2, '"Predicted Sentiment":', predictions[1])

```

For classification tasks, you can ask for predicted class-probabilities instead of predicted classes.


```{.python .input}
probs = predictor.predict_proba({'sentence': [sentence1, sentence2]})
print('"Sentence":', sentence1, '"Predicted Class-Probabilities":', probs[0])
print('"Sentence":', sentence2, '"Predicted Class-Probabilities":', probs[1])

```

## Sentence Similarity

Next, let's use AutoGluon to train a model for evaluating how semantically similar two sentences are. 
We use the [Semantic Textual Similarity Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) dataset for illustration.


```{.python .input}
train_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet')[['sentence1', 'sentence2', 'score']]
dev_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet')[['sentence1', 'sentence2', 'score']]
train_data.head(10)
```

In this data, the **score** column contains numerical values (which we'd like to predict) that are human-annotated similarity scores for each given pair of sentences.


```{.python .input}
print('Min score=', min(train_data['score']), ', Max score=', max(train_data['score']))
```

Let's train a regression model to predict these scores with `task.fit()`. 
Note that we only need to specify the label column and AutoGluon automatically determines the type of prediction problem and an appropriate loss function.
Once again, you should increase the short `time_limits` below to obtain reasonable performance in your own applications.


```{.python .input}
predictor_sts = task.fit(train_data, label='score',
                         time_limits='1min', ngpus_per_trial=1, seed=123,
                         output_directory='./ag_sts')
```

We again evaluate our trained model's performance on some separate test data. Below we choose to compute the following metrics: RMSE, Pearson Correlation, and Spearman Correlation.


```{.python .input}
dev_score = predictor_sts.evaluate(dev_data, metrics=['rmse', 'pearsonr', 'spearmanr'])
print('Best Config = {}'.format(predictor_sts.results['best_config']))
print('Total Time = {}s'.format(predictor_sts.results['total_time']))
print('RMSE = {:.2f}'.format(dev_score['rmse']))
print('PEARSONR = {:.4f}'.format(dev_score['pearsonr']))
print('SPEARMANR = {:.4f}'.format(dev_score['spearmanr']))
```

Let's use our model to predict the similarity score among these sentences:

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
Here, we demonstrate how to easily save and load a trained TextPrediction model.


```{.python .input}
predictor_sts.save('saved_dir')
predictor_sts_new = task.load('saved_dir')

score3 = predictor_sts_new.predict({'sentence1': [sentences[0]],
                                    'sentence2': [sentences[3]]})
print(score3)
```

## Extract Embeddings
:label:`sec_textprediction_extract_embedding`

After you have trained a predictor, you can also use the predictor to extract embeddings that maps the input data to a real vector. 
This can be useful for integrating with other AutoGluon modules like TabularPredictor. 
We can just feed the embeddings to TabularPredictor. 


```{.python .input}
embeddings = predictor_sts_new.extract_embedding(dev_data)
print(embeddings)
```

**Note:** `TextPrediction` depends on the [GluonNLP](https://gluon-nlp.mxnet.io/) package. 
Due to an ongoing upgrade of GluonNLP, we are currently using a custom version of the package: [autogluon-contrib-nlp](https://github.com/sxjscience/autogluon-contrib-nlp.git). In a future release, AutoGluon will switch to using the official GluonNLP, but the APIs demonstrated here will remain the same.
