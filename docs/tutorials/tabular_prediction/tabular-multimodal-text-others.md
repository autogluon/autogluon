# TabularPredictor for Multimodal Data Tables: Combining BERT and Classical Tabular Models

:label:`sec_tabularprediction_text_multimodal`

We will introduce how to use AutoGluon to deal with tabular data that involves text and categorical features.
This type of data, i.e., data which contains text and other features, is prevalent in real world applications.
For example, when building a sentiment analysis model of users' tweets, we can not only use the raw text in the 
tweets but also other features such as the topic of the tweet and the user profile. In the following, 
we will investigate different ways to ensemble the state-of-the-art (pretrained) language models in AutoGluon TextPrediction 
with all the other models used in AutoGluon's TabularPredictor. 
For more details about the inner-working of the neural network architecture used in AutoGluon TextPrediction, 
you may refer to Section ":ref:`sec_textprediction_architecture`" in :ref:`sec_textprediction_heterogeneous`.



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

train_df = pd.read_csv('product_sentiment_machine_hack/train.csv', index_col=0)
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
