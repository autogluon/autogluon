# Get started with AutoGluon for Multimodal Data: Text + Categorical

:label:`sec_tabularprediction_text_multimodal`

In this tutorial, we will introduce how to use AutoGluon to deal with multimodal data that involves text, categorical, and numerical features. These types of data are prevalent in real world applications. For example, when we try to analyze the sentiment of users' tweets, we can not only use the raw text in the tweets but also other features such as the topic of the tweet and the user profile. In the following, we explain different ways for combining a state-of-the-art pretrained language model for text data: [ELECTRA](https://arxiv.org/pdf/2003.10555.pdf) and the ensemble techniques in AutoGluon to improve the modeling performance for multimodal data.


```{.python .input}
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import autogluon
import pandas as pd
import pprint
import random
from autogluon.text import TextPrediction
from autogluon.tabular import TabularPrediction
import mxnet as mx

np.random.seed(123)
random.seed(123)
mx.random.seed(123)
```

## Product Sentiment Analysis Dataset

In this tutorial, we will use the product sentiment analysis dataset from this [MachineHack hackathon](https://www.machinehack.com/hackathons/product_sentiment_classification_weekend_hackathon_19/leaderboard). The goal of this task is to predict the user's sentiment towards a product given a review that is in raw text and the product's type, e.g., Tablet, Mobile, etc. We have split the original training data to be 90% for training and 10% for development.


```{.python .input}
!mkdir -p product_sentiment_machine_hack
!wget https://autogluon-text-data.s3.amazonaws.com/multimodal_text/machine_hack_product_sentiment/train.csv -O product_sentiment_machine_hack/train.csv
!wget https://autogluon-text-data.s3.amazonaws.com/multimodal_text/machine_hack_product_sentiment/dev.csv -O product_sentiment_machine_hack/dev.csv
!wget https://autogluon-text-data.s3.amazonaws.com/multimodal_text/machine_hack_product_sentiment/test.csv -O product_sentiment_machine_hack/test.csv
```


```{.python .input}
feature_columns = ['Product_Description', 'Product_Type']
label_column = 'Sentiment'

train_df = pd.read_csv('product_sentiment_machine_hack/train.csv')
dev_df = pd.read_csv('product_sentiment_machine_hack/dev.csv')
test_df = pd.read_csv('product_sentiment_machine_hack/test.csv')

train_df = train_df[feature_columns + [label_column]]
dev_df = dev_df[feature_columns + [label_column]]
test_df = test_df[feature_columns]
print('Number of training samples:', len(train_df))
print('Number of dev samples:', len(dev_df))
print('Number of test samples:', len(test_df))
```

There are two features in the dataset: the users' review about the product and the product's type. Also, there are four classes and we have split the train and dev set based on stratified sampling.


```{.python .input}
train_df
```


```{.python .input}
dev_df
```


```{.python .input}
test_df
```

## What can we get without mixing multiple data types?

First of all, let's try to train models without mixing the multi-modal data. We will use the TextPrediction model in AutoGluon to train a predictor with text data only. This will internally use the ELECTRA model as the backbone.


```{.python .input}
predictor_text_only = TextPrediction.fit(train_df[['Product_Description', 'Sentiment']],
                                         label=label_column,
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

We first train the baseline model in AutoGluon without the pretrained language model. Internally, AutoGluon uses the n-gram and TF-IDF based features.


```{.python .input}
predictor_model1 = TabularPrediction.fit(train_df,
                                         label=label_column,
                                         time_limits=None,
                                         eval_metric='accuracy',
                                         stopping_metric='accuracy',
                                         hyperparameters='default',
                                         output_directory='model1')
```


```{.python .input}
predictor_model1.leaderboard(dev_df)
```

We can find that combining the product type feature is quite essential for a good performance.

## Model 2: Extract Text Embedding and Use Tabular

The AutoGluon-Text offers the `extract_embedding()` functionality so we can try to use the trained text-only model to extract sentence embeddings and use AutoGluon to train the predictor.


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
predictor_model2 = TabularPrediction.fit(merged_train_data,
                                         label=label_column,
                                         time_limits=None,
                                         eval_metric='accuracy',
                                         stopping_metric='accuracy',
                                         hyperparameters='default',
                                         output_directory='model2')
```


```{.python .input}
predictor_model2.leaderboard(merged_dev_data)
```

## Model 3: Use the model in AutoGluon-Text


```{.python .input}
tabular_multimodel_hparam_v1 = {
    'GBM': [{}, {'extra_trees': True, 'AG_args': {'name_suffix': 'XT'}}],
    'CAT': {},
    'TEXT_NN_V1': {'AG_args': {'valid_stacker': False}},
}

predictor_model3 = TabularPrediction.fit(train_df,
                                      label=label_column,
                                      time_limits=None,
                                      eval_metric='accuracy',
                                      stopping_metric='accuracy',
                                      hyperparameters=tabular_multimodel_hparam_v1,
                                      output_directory='model3')
```


```{.python .input}
predictor_model3.leaderboard(dev_df)
```

## Model 4: Text Prediction + Stack Ensemble


```{.python .input}
predictor_model4 = TabularPrediction.fit(train_df,
                                         label=label_column,
                                         time_limits=None,
                                         eval_metric='accuracy',
                                         stopping_metric='accuracy',
                                         hyperparameters=tabular_multimodel_hparam_v1,
                                         num_bagging_folds=5,
                                         stack_ensemble_levels=1,
                                         output_directory='model4')
```


```{.python .input}
predictor_model4.leaderboard(dev_df)
```

## Model 5: Multimodal embedding + TabularPrediction


```{.python .input}
predictor_text_multimodal = TextPrediction.fit(train_df,
                                               label=label_column,
                                               time_limits=None,
                                               eval_metric='accuracy',
                                               stopping_metric='accuracy',
                                               hyperparameters='default_no_hpo',
                                               output_directory='predictor_text_multimodal')

train_sentence_multimodal_embeddings = predictor_text_multimodal.extract_embedding(train_df)
dev_sentence_multimodal_embeddings = predictor_text_multimodal.extract_embedding(dev_df)

predictor_model5 = TabularPrediction.fit(train_df.join(pd.DataFrame(train_sentence_multimodal_embeddings)),
                                         label=label_column,
                                         time_limits=None,
                                         eval_metric='accuracy',
                                         stopping_metric='accuracy',
                                         hyperparameters='default',
                                         output_directory='model5')
```


```{.python .input}
predictor_model5.leaderboard(dev_df.join(pd.DataFrame(dev_sentence_multimodal_embeddings)))
```

## Model 6: Use a larger backbone

Now, we will choose to use a larger backbone: ELECTRA-base. We will find that the performance gets improved after we change to use a larger backbone model.


```{.python .input}
from autogluon.text.text_prediction.text_prediction import ag_text_prediction_params
from autogluon.tabular.task.tabular_prediction.hyperparameter_configs import get_hyperparameter_config
import copy

text_nn_params = ag_text_prediction_params.create('default_no_hpo')
text_nn_params['models']['BertForTextPredictionBasic']['search_space']['model.backbone.name'] = 'google_electra_base'
text_nn_params['AG_args'] = {'valid_stacker': False}

tabular_multimodel_hparam_v2 = copy.deepcopy(tabular_multimodel_hparam_v1)
tabular_multimodel_hparam_v2['TEXT_NN_V1'] = text_nn_params

predictor_model6 = TabularPrediction.fit(train_df,
                                      label=label_column,
                                      time_limits=None,
                                      eval_metric='accuracy',
                                      stopping_metric='accuracy',
                                      hyperparameters=tabular_multimodel_hparam_v2,
                                      output_directory='model6')
```


```{.python .input}
predictor_model6.leaderboard(dev_df)
```
