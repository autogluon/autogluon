# AutoMM for Text - Quick Start
:label:`sec_automm_textprediction_beginner`

`MultiModalPredictor` can solve problems where the data are either image, text, numerical values, or categorical features. 
To get started, we first demonstrate how to use it to solve problems that only contain text. We pick two classical NLP problems for the purpose of demonstration:

- [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
- [Sentence Similarity](https://arxiv.org/abs/1910.03940)

Here, we format the NLP datasets as data tables where 
the feature columns contain text fields and the label column contain numerical (regression) / categorical (classification) values. 
Each row in the table corresponds to one training sample.

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
Given a new movie review, the goal is to predict the sentiment reflected in the text (in this case a **binary classification**, where reviews are 
labeled as 1 if they convey a positive opinion and labeled as 0 otherwise). Let's first load and look at the data, 
noting the labels are stored in a column called **label**.


```{.python .input}
from autogluon.core.utils.loaders import load_pd
train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
subsample_size = 1000  # subsample data for faster demo, try setting this to larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head(10)
```

Above the data happen to be stored in the [Parquet](https://databricks.com/glossary/what-is-parquet) format, but you can also directly `load()` data from a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file or other equivalent formats. 
While here we load files from [AWS S3 cloud storage](https://docs.aws.amazon.com/AmazonS3/latest/dev/Welcome.html), these could instead be local files on your machine. 
After loading, `train_data` is simply a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), 
where each row represents a different training example.

### Training

To ensure this tutorial runs quickly, we simply call `fit()` with a subset of 1000 training examples and limit its runtime to approximately 1 minute.
To achieve reasonable performance in your applications, you are recommended to set much longer `time_limit` (eg. 1 hour), or do not specify `time_limit` at all (`time_limit=None`).


```{.python .input}
from autogluon.multimodal import MultiModalPredictor
import uuid
model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor = MultiModalPredictor(label='label', eval_metric='acc', path=model_path)
predictor.fit(train_data, time_limit=180)
```

Above we specify that: the column named **label** contains the label values to predict, AutoGluon should optimize its predictions for the accuracy evaluation metric, 
trained models should be saved in the **automm_sst** folder, and training should run for around 60 seconds.

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
print('"Sentence":', sentence1, '"Predicted Sentiment":', predictions[0])
print('"Sentence":', sentence2, '"Predicted Sentiment":', predictions[1])
```

For classification tasks, you can ask for predicted class-probabilities instead of predicted classes.


```{.python .input}
probs = predictor.predict_proba({'sentence': [sentence1, sentence2]})
print('"Sentence":', sentence1, '"Predicted Class-Probabilities":', probs[0])
print('"Sentence":', sentence2, '"Predicted Class-Probabilities":', probs[1])
```

We can just as easily produce predictions over an entire dataset.


```{.python .input}
test_predictions = predictor.predict(test_data)
test_predictions.head()
```

### Save and Load

The trained predictor is automatically saved at the end of `fit()`, and you can easily reload it.

:::warning

`MultiModalPredictor.load()` used `pickle` module implicitly, which is known to be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. Never load data that could have come from an untrusted source, or that could have been tampered with. **Only load data you trust.**

:::

```{.python .input}
loaded_predictor = MultiModalPredictor.load(model_path)
loaded_predictor.predict_proba({'sentence': [sentence1, sentence2]})
```

You can also save the predictor to any location by calling `.save()`.


```{.python .input}
new_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
loaded_predictor.save(new_model_path)
loaded_predictor2 = MultiModalPredictor.load(new_model_path)
loaded_predictor2.predict_proba({'sentence': [sentence1, sentence2]})
```

### Extract Embeddings
:label:`sec_automm_textprediction_extract_embedding`

You can also use a trained predictor to extract embeddings that maps each row of the data table to an embedding vector extracted from intermediate neural network representations of the row.


```{.python .input}
embeddings = predictor.extract_embedding(test_data)
print(embeddings.shape)
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
:label:`sec_automm_textprediction_continuous_training`

You can also load a predictor and call `.fit()` again to continue training the same predictor with new data.


```{.python .input}
new_predictor = MultiModalPredictor.load(new_model_path)
new_predictor.fit(train_data, time_limit=30)
test_score = new_predictor.evaluate(test_data, metrics=['acc', 'f1'])
print(test_score)
```

## Sentence Similarity Task

Next, let's use MultiModalPredictor to train a model for evaluating how semantically similar two sentences are.
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
sts_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sts"
predictor_sts = MultiModalPredictor(label='score', path=sts_model_path)
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

Although the `MultiModalPredictor` currently supports classification and regression tasks, it can directly be used for 
many NLP tasks if you properly format them into a data table. Note that there can be many text columns in this data table. 
Refer to the [MultiModalPredictor documentation](../../api/autogluon.predictor.html#autogluon.multimodal.MultiModalPredictor.fit) to see all available methods/options.

Unlike `TabularPredictor` which trains/ensembles different types of models,
`MultiModalPredictor` focuses on selecting and finetuning deep learning based models. 
Internally, it integrates with [timm](https://github.com/rwightman/pytorch-image-models) , [huggingface/transformers](https://github.com/huggingface/transformers), 
[openai/clip](https://github.com/openai/CLIP) as the model zoo.

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
