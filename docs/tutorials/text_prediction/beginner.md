# Text Prediction - Quick Start
:label:`sec_textprediction_beginner`

Here we introduce the `TextPredictor`, which helps you automatically train and deploy models for various Natural Language Processing (NLP) problems.
This tutorial presents two examples to demonstrate how `TextPredictor` can be used for different NLP tasks including:

- [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
- [Sentence Similarity](https://arxiv.org/abs/1910.03940)

The general usage is similar to AutoGluon's `TabularPredictor`. We treat NLP datasets as tables where certain columns contain text fields and a special column contains the labels to predict. 
Here, the labels can be discrete categories (classification) or numerical values (regression).
`TextPredictor` fits neural networks to your data via transfer learning from pretrained NLP models like: [BERT](https://arxiv.org/pdf/1810.04805.pdf),
[ALBERT](https://arxiv.org/pdf/1909.11942.pdf), and [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB).
`TextPredictor` also enables training on multi-modal data tables that contain text, numeric and categorical columns, and can be used together with Hyperparameter Optimization (HPO), which will be introduced in the later section. In this part, we will explain the basic usage of `TextPredictor`.


```{.python .input}
%matplotlib inline

import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Example1: Sentiment Analysis

First, we consider the Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)) dataset, which consists of movie reviews and their associated sentiment. Given a new movie review, the goal is to predict the sentiment reflected in the text (in this case a **binary classification** problem, where reviews are labeled as 1 if they convey a positive opinion and labeled as 0 otherwise). Let's first load the data and view some examples, noting the labels are stored in a column called **label**.


```{.python .input}
from autogluon.core.utils.loaders.load_pd import load
train_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
dev_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
rand_idx = np.random.permutation(np.arange(len(train_data)))[:1000]
train_data = train_data.iloc[rand_idx]
train_data.head(10)
```

## Train Model

Above the data happen to be stored in a [Parquet](https://databricks.com/glossary/what-is-parquet) table format, but you can also directly `load()` data from a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file instead. While here we load files from [AWS S3 cloud storage](https://docs.aws.amazon.com/AmazonS3/latest/dev/Welcome.html), these could instead be local files on your machine. After loading, `train_data` is simply a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), where each row represents a different training example (for machine learning to be appropriate, the rows should be independent and identically distributed).

To ensure this tutorial runs quickly, we simply call `fit()` with a subset of 2000 training examples and limit its runtime to approximately 1 minute. 
To achieve reasonable performance in your applications, you are recommended to set much longer `time_limit` (eg. 1 hour), or do not specify `time_limit` at all (`time_limit=None`).


```{.python .input}
from autogluon.text import TextPredictor

predictor = TextPredictor(label='label', eval_metric='acc', path='./ag_sst')
predictor.fit(train_data, time_limit=60, seed=123) 
```

Above we specify that: the **label** column of our DataFrame contains the label-values to predict, AutoGluon should run for 60 seconds, each training run of an individual model (with particular hyperparameters) should run on 1 GPU, a particular random seed should be used to facilitate reproducibility, and that trained models should be saved in the **ag_sst** folder.

## Evaluation

Now you can use `predictor.evaluate()` to evaluate the trained model on a separate test data. By default, it will report the evaluation metric that you have specified, which is `accuracy` in our example.


```{.python .input}
dev_score = predictor.evaluate(dev_data)
print('Accuracy = {:.2f}%'.format(dev_score * 100))
```

If you'd like to evaluate on your other metrics, e.g., calculating the F1 score and the accuracy, you can specify the `metrics` argument when calling evaluate.


```{.python .input}
dev_score = predictor.evaluate(dev_data, metrics=['acc', 'f1'])
print(dev_score)
```

## Get predictions

And you can easily obtain predictions from these models by calling `predictor.predict()`.


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

## Explore the Intermediate Training Loss

Once the predictor has been trained, you can view the intermediate training results by fetching `predictor.results`.


```{.python .input}
predictor.results.tail(3)
```

## Save and Load
By default, we will automatically save the trained predictor once the fit is finished. You can load the trained model via the following code


```{.python .input}
loaded_predictor = TextPredictor.load('ag_sst')
loaded_predictor.predict_proba({'sentence': [sentence1, sentence2]})
```

You can also save the predictor to any location by calling `.save()`.


```{.python .input}
loaded_predictor.save('my_saved_dir')
loaded_predictor2 = TextPredictor.load('my_saved_dir')
loaded_predictor2.predict_proba({'sentence': [sentence1, sentence2]})
```

## Extract Embeddings
:label:`sec_textprediction_extract_embedding`

After you have trained a predictor, you can also use the predictor to extract embeddings that maps the input data to a real vector. 
This can be useful for integrating with other AutoGluon modules like TabularPredictor. 
We can just feed the embeddings to TabularPredictor.


```{.python .input}
embeddings = predictor.extract_embedding(dev_data)
print(embeddings)
```

Here, we use TSNE to visualize these extracted embeddings. We can see that there are two clusters.


```{.python .input}
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, random_state=123).fit_transform(embeddings)
for val, color in [(0, 'red'), (1, 'blue')]:
    idx = (dev_data['label'].to_numpy() == val).nonzero()
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], c=color, label=f'label={val}')
plt.legend(loc='best')
```

## Example2: Sentence Similarity

Next, let's use AutoGluon to train a model for evaluating how semantically similar two sentences are. 
We use the [Semantic Textual Similarity Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) dataset for illustration.


```{.python .input}
sts_train_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet')[['sentence1', 'sentence2', 'score']]
sts_dev_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet')[['sentence1', 'sentence2', 'score']]
sts_train_data.head(10)
```

In this data, the **score** column contains numerical values (which we'd like to predict) that are human-annotated similarity scores for each given pair of sentences.


```{.python .input}
print('Min score=', min(sts_train_data['score']), ', Max score=', max(sts_train_data['score']))
```

Let's train a regression model to predict these scores with `predictor.fit()`. 
Note that we only need to specify the label column and AutoGluon automatically determines the type of prediction problem and an appropriate loss function.
Once again, you should increase the short `time_limit` below to obtain reasonable performance in your own applications.


```{.python .input}
predictor_sts = TextPredictor(label='score', path='./ag_sts')
predictor_sts.fit(sts_train_data, time_limit=60, seed=123) 
```

We again evaluate our trained model's performance on some separate test data. Below we choose to compute the following metrics: RMSE, Pearson Correlation, and Spearman Correlation.


```{.python .input}
dev_score = predictor_sts.evaluate(sts_dev_data, metrics=['rmse', 'pearsonr', 'spearmanr'])
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
                                'sentence2': [sentences[1]]}, as_pandas=False)

score2 = predictor_sts.predict({'sentence1': [sentences[0]],
                                'sentence2': [sentences[2]]}, as_pandas=False)

score3 = predictor_sts.predict({'sentence1': [sentences[0]],
                                'sentence2': [sentences[3]]}, as_pandas=False)
print(score1, score2, score3)
```

**Note:** `TextPredictor` depends on the [GluonNLP](https://gluon-nlp.mxnet.io/) package. 
Due to an ongoing upgrade of GluonNLP, we are currently using a custom version of the package: [autogluon-contrib-nlp](https://github.com/sxjscience/autogluon-contrib-nlp.git). In a future release, AutoGluon will support the official GluonNLP 1.0, but the APIs demonstrated here will remain the same.
