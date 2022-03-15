# Text Prediction - Solving Multilingual Problems
:label:`sec_textprediction_multilingual`

People around the world speaks lots of languages. According to [SIL International](https://en.wikipedia.org/wiki/SIL_International)'s [Ethnologue: Languages of the World](https://en.wikipedia.org/wiki/Ethnologue), there are more than **7,100** spoken and signed languages. In fact, web data nowadays are highly multilingual and lots of real-world problems involve text written in languages other than English.

In this tutorial, we introduce how AutoGluon Text can help you build multilingual models. For the purpose of demonstration, we use the [Cross-Lingual Amazon Product Review Sentiment](https://webis.de/data/webis-cls-10.html) dataset, which comprises about 800,000 Amazon product reviews in four languages: English, German, French, and Japanese. We will demonstrate how to use AutoGluon Text to build sentiment classification models on the German fold of this dataset in two ways:

- Finetune the German BERT
- Cross-lingual transfer from English to German

## Load Dataset

The [Cross-Lingual Amazon Product Review Sentiment](https://webis.de/data/webis-cls-10.html) dataset contains Amazon product reviews in four languages. Here, we load the English and German fold of the dataset. In the label column, `0` means negative sentiment and `1` means positive sentiment.


```{.python .input}
!wget https://automl-mm-bench.s3.amazonaws.com/multilingual-datasets/amazon_review_sentiment_cross_lingual.zip -O amazon_review_sentiment_cross_lingual.zip
!unzip -o amazon_review_sentiment_cross_lingual.zip -d .
```


```{.python .input}
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

train_de_df = pd.read_csv('amazon_review_sentiment_cross_lingual/de_train.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
                .sample(1000, random_state=123)
train_de_df.reset_index(inplace=True, drop=True)

test_de_df = pd.read_csv('amazon_review_sentiment_cross_lingual/de_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123)
test_de_df.reset_index(inplace=True, drop=True)
print(train_de_df)
```


```{.python .input}
train_en_df = pd.read_csv('amazon_review_sentiment_cross_lingual/en_train.tsv',
                          sep='\t',
                          header=None,
                          names=['label', 'text']) \
                .sample(1000, random_state=123)
train_en_df.reset_index(inplace=True, drop=True)

test_en_df = pd.read_csv('amazon_review_sentiment_cross_lingual/en_test.tsv',
                          sep='\t',
                          header=None,
                          names=['label', 'text']) \
               .sample(200, random_state=123)
test_en_df.reset_index(inplace=True, drop=True)
print(train_en_df)
```

## Finetune the German BERT

Our first approach is to finetune the [German BERT model](https://www.deepset.ai/german-bert) pretrained by deepset. Since AutoGluon Text integrates with the [Huggingface/Transformers](https://huggingface.co/docs/transformers/index) (as explained in :ref:`sec_textprediction_customization`), we directly load the German BERT model via Huggingface/Transformers, with the key as [bert-base-german-cased](https://huggingface.co/bert-base-german-cased). To simplify the experiment, we also just finetune for 4 epochs.


```{.python .input}
from autogluon.text import TextPredictor

predictor = TextPredictor(label='label')
predictor.fit(train_de_df,
              hyperparameters={
                  'model.hf_text.checkpoint_name': 'bert-base-german-cased',
                  'optimization.max_epochs': 4
              })
```


```{.python .input}
score = predictor.evaluate(test_de_df)
print('Score on the German Testset:')
print(score)
```


```{.python .input}
score = predictor.evaluate(test_en_df)
print('Score on the English Testset:')
print(score)
```

We can find that the model can achieve good performance on the German dataset but performs poorly on the English dataset. Next, we will show how to enable cross-lingual transfer so you can get a model that can magically work for **both German and English**.

## Cross-lingual Transfer

In the real-world scenario, it is pretty common that you have trained a model for English and would like to extend the model to support other languages like German. This setting is also known as cross-lingual transfer. 
One way to solve the problem is to apply a machine translation model to translate the sentences from the other language (e.g., German) to English and apply the English model.
However, as showed in ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/pdf/1911.02116.pdf), there is a better and cost-friendlier way for cross lingual transfer, enabled via large-scale multilingual pretraining.
The author showed that via large-scale pretraining, the backbone (called XLM-R) is able to conduct *zero-shot* cross lingual transfer, meaning that you can directly apply the model trained in the English dataset to datasets in other languages. 
It also outperforms the baseline "TRANSLATE-TEST", meaning to translate the data from other languages to English and apply the English model. 

In AutoGluon, you can just turn on `presets="multilingual"` to load a backbone that is suitable for zero-shot transfer. 
Internally, we will automatically use state-of-the-art models like [DeBERTa-V3](https://arxiv.org/abs/2111.09543).


```{.python .input}
from autogluon.text import TextPredictor

predictor = TextPredictor(label='label')
predictor.fit(train_en_df,
              presets='multilingual',
              hyperparameters={
                  'optimization.max_epochs': 4
              })
```


```{.python .input}
score_in_en = predictor.evaluate(test_en_df)
print('Score in the English Testset:')
print(score_in_en)
```


```{.python .input}
score_in_de = predictor.evaluate(test_de_df)
print('Score in the German Testset:')
print(score_in_de)
```

We can see that the model works for both German and English!

Let's also inspect the model's performance on Japanese:

```{.python .input}
test_jp_df = pd.read_csv('amazon_review_sentiment_cross_lingual/jp_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123)
test_jp_df.reset_index(inplace=True, drop=True)
print(test_jp_df)
```

```{.python .input}
print('Negative labe ratio of the Japanese Testset=', test_jp_df['label'].value_counts()[0] / len(test_jp_df))
score_in_jp = predictor.evaluate(test_jp_df)
print('Score in the Japanese Testset:')
print(score_in_jp)
```
Amazingly, the model also works for Japanese!
