# Text Prediction - Customization (For MXNet backend only)
:label:`sec_textprediction_customization`

This advanced tutorial teaches you how to customize the hyperparameters in `TextPredictor` by specifying:

- A custom search space of candidate hyperparameter values to consider.
- Which hyperparameter optimization (HPO) method should be used to actually search through this space.


```{.python .input}
import numpy as np
import warnings
import autogluon as ag
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Stanford Sentiment Treebank Data

For demonstration, we use the Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)) dataset.


```{.python .input}
from autogluon.core import TabularDataset
subsample_size = 1000  # subsample for faster demo, you may try specifying larger value
train_data = TabularDataset('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = TabularDataset('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head(10)
```

## Configuring the TextPredictor

### Pre-configured Hyperparameters

We provided a series of pre-configured hyperparameters. You may list the keys from `ag_text_presets` via `list_presets`.


```{.python .input}
from autogluon.text.text_prediction.legacy_presets import ag_text_presets, list_presets
list_presets()
```

There are two kinds of presets. The `simple_presets` are pre-defined configurations recommended for most users, which allow you specify whether you care more about predictive accuracy (`'best_quality'`) or more about training/inference speed (`'lower_quality_fast_train'`)

The `advanced_presets` are pre-configured networks using different Transformer backbones such as ELECTRA, RoBERTa, or Multilingual BERT, and different feature fusion strategies. For example, `electra_small_fuse_late` means we use the ELECTRA-small model as the network backbone for text fields  and use the late fusion strategy described in ":ref:`sec_textprediction_architecture`". The  `default` preset is the same as `electra_base_fuse_late`. Now let's train a model on our data with specified `presets`.


```{.python .input}
from autogluon.text import TextPredictor
predictor = TextPredictor(path='ag_text_sst_electra_small', eval_metric='acc', label='label', backend='mxnet')
predictor.set_verbosity(0)
predictor.fit(train_data, presets='electra_small_fuse_late', time_limit=60, seed=123)
```

Below we report both `f1` and `acc` metrics for our predictions. Note that if you really want to obtain the best F1 score, you should set `eval_metric='f1'` when constructing the TextPredictor.


```{.python .input}
predictor.evaluate(test_data, metrics=['f1', 'acc'])
```

To view the pre-registered hyperparameters, you can call `ag_text_presets.create(presets_name)`, e.g.,


```{.python .input}
import pprint
pprint.pprint(ag_text_presets.create('electra_small_fuse_late'))
```

Another way to specify a custom TextPredictor configuration is via the `hyperparameters` argument.


```{.python .input}
predictor = TextPredictor(path='ag_text_customize1', eval_metric='acc', label='label', backend='mxnet')
predictor.fit(train_data, hyperparameters=ag_text_presets.create('electra_small_fuse_late'),
              time_limit=30, seed=123)
```

### Custom Hyperparameter Values

The pre-registered configurations provide reasonable default hyperparameters. A common workflow is to first train a model with one of the presets and then tune some hyperparameters to see if the performance can be further improved. In the example below, we set the number of training epochs to 5 and the learning rate to be 5E-5.


```{.python .input}
hyperparameters = ag_text_presets.create('electra_small_fuse_late')
hyperparameters['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = 5
hyperparameters['models']['MultimodalTextModel']['search_space']['optimization.lr'] = ag.core.space.Categorical(5E-5)

predictor = TextPredictor(path='ag_text_customize2', eval_metric='acc', label='label', backend='mxnet')
predictor.fit(train_data, hyperparameters=hyperparameters, time_limit=30, seed=123)
```

### Register Your Own Configuration

You can also register your custom hyperparameter settings as new presets in `ag_text_presets`. Below, the `electra_small_fuse_late_train5` preset uses ELECTRA-small as its backbone
and trains for 5 epochs with a weight-decay of 1E-2.


```{.python .input}
@ag_text_presets.register()
def electra_small_fuse_late_train5():
    hyperparameters = ag_text_presets.create('electra_small_fuse_late')
    hyperparameters['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = 5
    hyperparameters['models']['MultimodalTextModel']['search_space']['optimization.wd'] = 1E-2
    return hyperparameters

predictor = TextPredictor(path='ag_text_customize3', eval_metric='acc', label='label', backend='mxnet')
predictor.fit(train_data, presets='electra_small_fuse_late_train5', time_limit=60, seed=123)
```

## HPO over a Customized Search Space via Bayesian Optimization

To control which hyperparameter values are considered during `fit()`, we specify the `hyperparameters` argument. Rather than specifying a particular fixed value for a hyperparameter, we can specify a space of values to search over via `ag.core.space`. 
We can also specify which HPO method to use for the search via `search_strategy`. 
By default, we will use [Bayesian Optimization](https://arxiv.org/pdf/1807.02811.pdf) as the searcher.
In this example, we search for good values of the following hyperparameters:

- warmup
- number of hidden units in the final MLP layer that maps aggregated features to output prediction
- learning rate
- weight decay


```{.python .input}
def electra_small_basic_demo_hpo():
    hparams = ag_text_presets.create('electra_small_fuse_late')
    search_space = hparams['models']['MultimodalTextModel']['search_space']
    search_space['optimization.per_device_batch_size'] = 8
    search_space['model.network.agg_net.mid_units'] = ag.core.space.Int(32, 128)
    search_space['optimization.warmup_portion'] = ag.core.space.Categorical(0.1, 0.2)
    search_space['optimization.lr'] = ag.core.space.Real(1E-5, 2E-4)
    search_space['optimization.wd'] = ag.core.space.Categorical(1E-4, 1E-3, 1E-2)
    search_space['optimization.num_train_epochs'] = 5
    return hparams
```

We can now call `fit()` with hyperparameter-tuning over our custom search space.
Below `num_trials` controls the maximal number of different hyperparameter configurations for which AutoGluon will train models (4 models are trained under different hyperparameter configurations in this case). To achieve good performance in your applications, you should use larger values of `num_trials`, which may identify superior hyperparameter values but will require longer runtimes.


```{.python .input}
predictor_sst_rs = TextPredictor(path='ag_text_sst_random_search', label='label', eval_metric='acc', backend='mxnet')
predictor_sst_rs.set_verbosity(0)
predictor_sst_rs.fit(train_data,
                      hyperparameters=electra_small_basic_demo_hpo(),
                      time_limit=60 * 2,
                      num_trials=4,
                      seed=123)
```

We can again evaluate our model's performance on separate test data.


```{.python .input}
test_score = predictor_sst_rs.evaluate(test_data, metrics=['acc', 'f1'])
print('Best Config = {}'.format(predictor_sst_rs.results['best_config']))
print('Total Time = {}s'.format(predictor_sst_rs.results['total_time']))
print('Accuracy = {:.2f}%'.format(test_score['acc'] * 100))
print('F1 = {:.2f}%'.format(test_score['f1'] * 100))
```
