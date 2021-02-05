# Explore Models for Data Tables with Text and Categorical Features

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
from autogluon.text import TextPrediction
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

train_df = pd.read_csv('product_sentiment_machine_hack/train.csv')
dev_df = pd.read_csv('product_sentiment_machine_hack/dev.csv')
test_df = pd.read_csv('product_sentiment_machine_hack/test.csv')

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
train_df
```


```{.python .input}
dev_df
```


```{.python .input}
test_df
```

## What happens if we ignore all the non-text features?

First of all, let's try to ignore all the non-text features. We will use the TextPrediction model 
in AutoGluon to train a predictor with text data only. This will internally use the ELECTRA-small 
model as the backbone. As we can see, the result is not very good.


```{.python .input}
predictor_text_only = TextPrediction.fit(train_df[['Product_Description', 'Sentiment']],
                                         label=label,
                                         time_limits=None,
                                         ngpus_per_trial=1,
                                         hyperparameters='default_no_hpo',
                                         eval_metric='accuracy',
                                         stopping_metric='accuracy',
                                         output_directory='ag_text_only')
```


```{.python .input}
print(predictor_text_only.evaluate(dev_df[['Product_Description', 'Sentiment']], metrics='accuracy'))
```

## Model 1:  Baseline with N-Gram + TF-IDF

The first baseline model is to directly call AutoGluon's TabularPredictor to train a predictor.
TabularPredictor uses the n-gram and TF-IDF based features for text columns and considers 
text and categorical columns simultaneously.

```{.python .input}
predictor_model1 = TabularPredictor(label=label, eval_metric='accuracy', path='model1').fit(train_df)
```


```{.python .input}
predictor_model1.leaderboard(dev_df, silent=True)
```

We can find that using product type (a categorical column) is quite essential for good performance in this task. 
The accuracy is much higher than the model trained with only text column. 

## Model 2: Extract Text Embedding and Use Tabular Predictor

Our second attempt in combining text and other features is to use the trained TextPrediction model to extract embeddings and 
use TabularPredictor to build the predictor on top of the text embeddings. 
The AutoGluon TextPrediction model offers the `extract_embedding()` functionality (For more details, go to :ref:`sec_textprediction_extract_embedding`), 
so we are able to build a two-stage model. In the first stage, we use the text-only model to extract sentence embeddings. 
In the second stage, we use TabularPredictor to get the final model.


```{.python .input}
train_sentence_embeddings = predictor_text_only.extract_embedding(train_df)
dev_sentence_embeddings = predictor_text_only.extract_embedding(dev_df)
print(train_sentence_embeddings)
```


```{.python .input}
merged_train_data = train_df.join(pd.DataFrame(train_sentence_embeddings))
merged_dev_data = dev_df.join(pd.DataFrame(dev_sentence_embeddings))
print(merged_train_data)
```


```{.python .input}
predictor_model2 = TabularPredictor(label=label, eval_metric='accuracy', path='model2').fit(merged_train_data)
```


```{.python .input}
predictor_model2.leaderboard(merged_dev_data, silent=True)
```

The performance is better than the first model.

## Model 3: Use the Neural Network in AutoGluon-Text in Tabular Weighted Ensemble

Another option is to directly include the neural network in AutoGluon-Text as one candidate of TabularPredictor. We can do that now by changing the hyperparameters. Note that for the purpose of this tutorial, we are manually setting the `hyperparameters` and we will release some good pre-configurations soon.


```{.python .input}
tabular_multimodel_hparam_v1 = {
    'GBM': [{}, {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}],
    'CAT': {},
    'TEXT_NN_V1': {},
}

predictor_model3 = TabularPredictor(label=label, eval_metric='accuracy', path='model3').fit(
    train_df, hyperparameters=tabular_multimodel_hparam_v1
)
```


```{.python .input}
predictor_model3.leaderboard(dev_df, silent=True)
```

## Model 4: K-Fold Bagging and Stack Ensemble

A more advanced strategy is to use 5-fold bagging and call stack ensembling. This is expected to improve the final performance.


```{.python .input}
predictor_model4 = TabularPredictor(label=label, eval_metric='accuracy', path='model4').fit(
    train_df, hyperparameters=tabular_multimodel_hparam_v1, num_bag_folds=5, num_stack_levels=1
)
```


```{.python .input}
predictor_model4.leaderboard(dev_df, silent=True)
```

## Model 5: Multimodal embedding + TabularPredictor

Also, since the neural network in text prediction can directly handle multi-modal data, we can fit a model with TextPrediction first and then use that as an embedding extractor. This can be viewed as an improved version of Model-2.


```{.python .input}
predictor_text_multimodal = TextPrediction.fit(train_df,
                                               label=label,
                                               time_limits=None,
                                               eval_metric='accuracy',
                                               stopping_metric='accuracy',
                                               hyperparameters='default_no_hpo',
                                               output_directory='predictor_text_multimodal')

train_sentence_multimodal_embeddings = predictor_text_multimodal.extract_embedding(train_df)
dev_sentence_multimodal_embeddings = predictor_text_multimodal.extract_embedding(dev_df)

predictor_model5 = TabularPredictor(label=label, eval_metric='accuracy', path='model5').fit(train_df)
```


```{.python .input}
predictor_model5.leaderboard(dev_df.join(pd.DataFrame(dev_sentence_multimodal_embeddings)), silent=True)
```

## Model 6: Use a larger backbone

Now, we will choose to use a larger backbone: ELECTRA-base. We will find that the performance gets improved after we change to use a larger backbone model. 
However, we should notice that the training time will be longer and the inference cost will be higher.


```{.python .input}
from autogluon.text.text_prediction.text_prediction import ag_text_prediction_params
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
import copy

text_nn_params = ag_text_prediction_params.create('default_electra_base_no_hpo')

tabular_multimodel_hparam_v2 = {
    'GBM': [{}, {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}],
    'CAT': {},
    'TEXT_NN_V1': text_nn_params,
}

predictor_model6 = TabularPredictor(label=label, eval_metric='accuracy', path='model6').fit(
    train_df, hyperparameters=tabular_multimodel_hparam_v2
)
```


```{.python .input}
predictor_model6.leaderboard(dev_df, silent=True)
```

## Major Takeaways

After performing these comparisons, we have the following takeaways:

- The multimodal text neural network structure used in TextPrediction is a good for dealing with tabular data with text and categorical features.

- K-fold bagging / stacking and weighted ensemble are helpful

- We need a larger backbone. This aligns with the observation in recent papers, e.g., [Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/abs/2010.14701).
