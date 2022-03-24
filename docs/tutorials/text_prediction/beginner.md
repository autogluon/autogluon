# Text Prediction - Quick Start
:label:`sec_textprediction_beginner`

Here we briefly demonstrate the `TextPredictor`, which helps you automatically train and deploy models for various Natural Language Processing (NLP) tasks.
This tutorial presents two examples of NLP tasks:

- [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
- [Sentence Similarity](https://arxiv.org/abs/1910.03940)

The general usage of the `TextPredictor` is similar to AutoGluon's `TabularPredictor`. We format NLP datasets as tables where certain columns contain text fields and a special column contains the labels to predict, and each row corresponds to one training example.
Here, the labels can be discrete categories (classification) or numerical values (regression). In fact, `TextPredictor` also enables training on multi-modal data tables that contain text, numeric and categorical columns
and also support solving multilingual problems. You may refer to multimodal / multilingual usage in :ref:`sec_textprediction_multimodal` and :ref:`sec_textprediction_multilingual`.


```{.python .input}
%matplotlib inline

import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Sentiment Analysis Task

First, we consider the Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)) dataset, which consists of movie reviews and their associated sentiment. 
Given a new movie review, the goal is to predict the sentiment reflected in the text (in this case a **binary classification**, where reviews are labeled as 1 if they convey a positive opinion and labeled as 0 otherwise). Let's first load and look at the data, noting the labels are stored in a column called **label**.


```{.python .input}
from autogluon.core.utils.loaders import load_pd
train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
subsample_size = 1000  # subsample data for faster demo, try setting this to larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head(10)
```

Above the data happen to be stored in a [Parquet](https://databricks.com/glossary/what-is-parquet) table format, but you can also directly `load()` data from a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file instead. While here we load files from [AWS S3 cloud storage](https://docs.aws.amazon.com/AmazonS3/latest/dev/Welcome.html), these could instead be local files on your machine. After loading, `train_data` is simply a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), where each row represents a different training example (for machine learning to be appropriate, the rows should be independent and identically distributed).

### Training

To ensure this tutorial runs quickly, we simply call `fit()` with a subset of 1000 training examples and limit its runtime to approximately 1 minute.
To achieve reasonable performance in your applications, you are recommended to set much longer `time_limit` (eg. 1 hour), or do not specify `time_limit` at all (`time_limit=None`).


```{.python .input}
from autogluon.text import TextPredictor

predictor = TextPredictor(label='label', eval_metric='acc', path='./ag_sst')
predictor.fit(train_data, time_limit=60)
```

Above we specify that: the column named **label** contains the label values to predict, AutoGluon should optimize its predictions for the accuracy evaluation metric,  trained models should be saved in the **ag_sst** folder, and training should run for around 60 seconds.

### Evaluation

After training, we can easily evaluate our predictor on separate test data formatted similarly to our training data.


```{.python .input}
test_score = predictor.evaluate(test_data)
print(test_score)
```

By default, `evaluate()` will report the evaluation metric previously specified, which is `accuracy` in our example. You may also specify additional metrics, e.g. F1 score, when calling evaluate.


```{.python .input}
test_score = predictor.evaluate(test_data, metrics=['acc', 'f1'])
print(test_score)
```

### Prediction

And you can easily obtain predictions from these models by calling `predictor.predict()`.


```{.python .input}
sentence1 = "it's a charming and often affecting journey."
sentence2 = "It's slow, very, very, very slow."
predictions = predictor.predict({'sentence': [sentence1, sentence2]})
print('"Sentence":', sentence1, '"Predicted Sentiment":', predictions.iloc[0])
print('"Sentence":', sentence2, '"Predicted Sentiment":', predictions.iloc[1])
```

For classification tasks, you can ask for predicted class-probabilities instead of predicted classes.


```{.python .input}
probs = predictor.predict_proba({'sentence': [sentence1, sentence2]})
print('"Sentence":', sentence1, '"Predicted Class-Probabilities":', probs.iloc[0])
print('"Sentence":', sentence2, '"Predicted Class-Probabilities":', probs.iloc[1])
```

We can just as easily produce predictions over an entire dataset.


```{.python .input}
test_predictions = predictor.predict(test_data)
test_predictions.head()
```

### Save and Load

The trained predictor is automatically saved at the end of `fit()`, and you can easily reload it.


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

### Extract Embeddings
:label:`sec_textprediction_extract_embedding`

You can also use a trained predictor to extract embeddings that maps each row of the data table to an embedding vector extracted from intermediate neural network representations of the row.


```{.python .input}
embeddings = predictor.extract_embedding(test_data)
print(embeddings)
```

Here, we use TSNE to visualize these extracted embeddings. We can see that there are two clusters corresponding to our two labels, since this network has been trained to discriminate between these labels.


```{.python .input}
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, random_state=123).fit_transform(embeddings)
for val, color in [(0, 'red'), (1, 'blue')]:
    idx = (test_data['label'].to_numpy() == val).nonzero()
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], c=color, label=f'label={val}')
plt.legend(loc='best')
```

### Continuous Training
:label:`sec_textprediction_continuous_training`

You can also load a predictor and call `.fit()` again to continue training the same predictor with new data.


```{.python .input}
new_predictor = TextPredictor.load('ag_sst')
new_predictor.fit(train_data, time_limit=30, save_path='ag_sst_continue_train')
test_score = new_predictor.evaluate(test_data, metrics=['acc', 'f1'])
print(test_score)
```

## Sentence Similarity Task

Next, let's use AutoGluon to train a model for evaluating how semantically similar two sentences are.
We use the [Semantic Textual Similarity Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) dataset for illustration.


```{.python .input}
sts_train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet')[['sentence1', 'sentence2', 'score']]
sts_test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet')[['sentence1', 'sentence2', 'score']]
sts_train_data.head(10)
```

In this data, the column named **score** contains numerical values (which we'd like to predict) that are human-annotated similarity scores for each given pair of sentences.


```{.python .input}
print('Min score=', min(sts_train_data['score']), ', Max score=', max(sts_train_data['score']))
```

Let's train a regression model to predict these scores. Note that we only need to specify the label column and AutoGluon automatically determines the type of prediction problem and an appropriate loss function. Once again, you should increase the short `time_limit` below to obtain reasonable performance in your own applications.


```{.python .input}
predictor_sts = TextPredictor(label='score', path='./ag_sts')
predictor_sts.fit(sts_train_data, time_limit=60)
```

We again evaluate our trained model's performance on separate test data. Below we choose to compute the following metrics: RMSE, Pearson Correlation, and Spearman Correlation.


```{.python .input}
test_score = predictor_sts.evaluate(sts_test_data, metrics=['rmse', 'pearsonr', 'spearmanr'])
print('RMSE = {:.2f}'.format(test_score['rmse']))
print('PEARSONR = {:.4f}'.format(test_score['pearsonr']))
print('SPEARMANR = {:.4f}'.format(test_score['spearmanr']))
```

Let's use our model to predict the similarity score between a few sentences.


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

Although the `TextPredictor` currently supports classification and regression tasks, it can directly be used for 
many NLP tasks if you properly format them into a data table. Note that there can be many text columns in this data table. 
Refer to the [TextPredictor documentation](../../api/autogluon.predictor.html#autogluon.text.TextPredictor.fit) to see all available methods/options.

Unlike `TabularPredictor` which trains/ensembles many different types of models,
`TextPredictor` focuses on fine-tuning deep learning based models. It supports transfer learning from pretrained NLP models like: [BERT](https://arxiv.org/pdf/1810.04805.pdf),
[ALBERT](https://arxiv.org/pdf/1909.11942.pdf), and [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB).

**Note:** `TextPredictor` uses `pytorch` as the default backend.
