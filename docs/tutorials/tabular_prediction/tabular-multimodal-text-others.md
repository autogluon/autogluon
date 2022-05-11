# Multimodal Data Tables: Combining BERT/Transformers and Classical Tabular Models

:label:`sec_tabularprediction_text_multimodal`

**Tip**: If your data contains images, consider also checking out :ref:`sec_tabularprediction_multimodal` which handles images in addition to text and tabular features.

Here we introduce how to use AutoGluon Tabular to deal with multimodal tabular data that contains text, numeric, and categorical columns. In AutoGluon, **raw text data** is considered as a first-class citizen of data tables. AutoGluon Tabular can help you train and combine a diverse set of models including classical tabular models like LightGBM/RF/CatBoost as well as our pretrained NLP model based multimodal network that is introduced in Section ":ref:`sec_textprediction_architecture`" of :ref:`sec_textprediction_multimodal` (used by AutoGluon's `TextPredictor`).


```{.python .input}
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint
import random
from autogluon.tabular import TabularPredictor

np.random.seed(123)
random.seed(123)
```

## Product Sentiment Analysis Dataset

We consider the product sentiment analysis dataset from a [MachineHack hackathon](https://www.machinehack.com/hackathons/product_sentiment_classification_weekend_hackathon_19/leaderboard). The goal is to predict a user's sentiment towards a product given their review (raw text) and a categorical feature indicating the product's type (e.g., Tablet, Mobile, etc.). We have already split the original dataset to be 90% for training and 10% for development/testing (if submitting your models to the hackathon, we recommend training them on 100% of the dataset).

```{.python .input}
!mkdir -p product_sentiment_machine_hack
!wget https://autogluon-text-data.s3.amazonaws.com/multimodal_text/machine_hack_product_sentiment/train.csv -O product_sentiment_machine_hack/train.csv
!wget https://autogluon-text-data.s3.amazonaws.com/multimodal_text/machine_hack_product_sentiment/dev.csv -O product_sentiment_machine_hack/dev.csv
!wget https://autogluon-text-data.s3.amazonaws.com/multimodal_text/machine_hack_product_sentiment/test.csv -O product_sentiment_machine_hack/test.csv
```

```{.python .input}
subsample_size = 2000  # for quick demo, try setting to larger values
feature_columns = ['Product_Description', 'Product_Type']
label = 'Sentiment'

train_df = pd.read_csv('product_sentiment_machine_hack/train.csv', index_col=0).sample(2000, random_state=123)
dev_df = pd.read_csv('product_sentiment_machine_hack/dev.csv', index_col=0)
test_df = pd.read_csv('product_sentiment_machine_hack/test.csv', index_col=0)

train_df = train_df[feature_columns + [label]]
dev_df = dev_df[feature_columns + [label]]
test_df = test_df[feature_columns]
print('Number of training samples:', len(train_df))
print('Number of dev samples:', len(dev_df))
print('Number of test samples:', len(test_df))
```

There are two features in the dataset: the users' review of the product and the product's type, and four possible classes to predict.

```{.python .input}
train_df.head()
```

```{.python .input}
dev_df.head()
```

```{.python .input}
test_df.head()
```

## AutoGluon Tabular with Multimodal Support

To utilize the `TextPredictor` model inside of `TabularPredictor`, we must specify the `hyperparameters = 'multimodal'` in AutoGluon Tabular. Internally, this will train multiple tabular models as well as the TextPredictor model, and then combine them via either a weighted ensemble or stack ensemble, as  explained in [AutoGluon Tabular Paper](https://arxiv.org/pdf/2003.06505.pdf). If you do not specify `hyperparameters = 'multimodal'`, then AutoGluon Tabular will simply featurize text fields using N-grams and train only tabular models (which may work better if your text is mostly uncommon strings/vocabulary).


```{.python .input}
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='Sentiment', path='ag_tabular_product_sentiment_multimodal')
predictor.fit(train_df, hyperparameters='multimodal')
```

```{.python .input}
predictor.leaderboard(dev_df)
```

## Improve the Performance with Stack Ensemble

You can improve predictive performance by using stack ensembling. One way to turn it on is as follows:

```
predictor.fit(train_df, hyperparameters='multimodal', num_bag_folds=5, num_stack_levels=1)
```

or using:

```
predictor.fit(train_df, hyperparameters='multimodal', presets='best_quality')
```

which will automatically select values for `num_stack_levels` (how many stacking layers) and `num_bag_folds` (how many folds to split data into during bagging).
Stack ensembling can take much longer, so we won't run with this configuration here. You may explore more examples in https://github.com/awslabs/autogluon/tree/master/examples/text_prediction, which demonstrate how you can achieve top performance in competitions with a stack ensembling based solution.
