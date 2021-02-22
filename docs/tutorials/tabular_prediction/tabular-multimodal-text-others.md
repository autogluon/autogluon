# TabularPredictor for Multimodal Data Tables: Combining BERT and Classical Tabular Models

:label:`sec_tabularprediction_text_multimodal`

We will introduce how to use AutoGluon Tabular to deal with multimodal tabular data that involves text, numeric, and categorical features. This type of data is prevalent in real world applications. In AutoGluon, **raw text data** is considered as a first-class citizen of data tables. AutoGluon Tabular can help you train and combine a diverse set of models including the classical ones like LightGBM/CatBoost and the more recent Pretrained Language Model (PLM) based multimodal network that is introduced in Section ":ref:`sec_textprediction_architecture`" of :ref:`sec_textprediction_multimodal`.


```{.python .input}
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint
import random
from autogluon.tabular import TabularPredictor
import mxnet as mx

np.random.seed(123)
random.seed(123)
mx.random.seed(123)
```

## Product Sentiment Analysis Dataset

In the following, we will use the product sentiment analysis dataset from this [MachineHack hackathon](https://www.machinehack.com/hackathons/product_sentiment_classification_weekend_hackathon_19/leaderboard). The goal of this task is to predict the user's sentiment towards a product given a review that is in raw text and the product's type, e.g., Tablet, Mobile, etc. We have split the original training data to be 90% for training and 10% for development.


```{.python .input}
!mkdir -p product_sentiment_machine_hack
!wget https://autogluon-text-data.s3.amazonaws.com/multimodal_text/machine_hack_product_sentiment/train.csv -O product_sentiment_machine_hack/train.csv
!wget https://autogluon-text-data.s3.amazonaws.com/multimodal_text/machine_hack_product_sentiment/dev.csv -O product_sentiment_machine_hack/dev.csv
!wget https://autogluon-text-data.s3.amazonaws.com/multimodal_text/machine_hack_product_sentiment/test.csv -O product_sentiment_machine_hack/test.csv
```


```{.python .input}
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

There are two features in the dataset: the users' review of the product and the product's type. 
Also, there are four classes and we have split the train and dev set based on stratified sampling.


```{.python .input}
train_df.head(3)
```


```{.python .input}
dev_df.head(3)
```


```{.python .input}
test_df.head(3)
```

## AutoGluon Tabular with Multimodal Support

We can directly specify the modeling hyperparameters as `multimodal` in AutoGluon Tabular. Internally, it will train multiple models and combine them via either weighted ensemble, or the stack ensembling, which is explained in [AutoGluon Tabular Paper](https://arxiv.org/pdf/2003.06505.pdf).


```{.python .input}
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='Sentiment', path='ag_tabular_product_sentiment_multimodal')
predictor.fit(train_df, hyperparameters='multimodal')
```


```{.python .input}
predictor.leaderboard(dev_df)
```

## Improve the Performance with Stack Ensemble

You can improve the performance by using stack ensembling. One way to turn it on is to call `predictor.fit(train_df, hyperparameters='multimodal', num_bag_folds=5, num_stack_levels=1)`. Due to the time constraint of tutorials, we won't run with this configuration and you may check more examples in https://github.com/awslabs/autogluon/tree/master/examples/text_prediction, where you can achieve top performance in competitions with the stack ensembling based solution.
