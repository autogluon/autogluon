# Text Prediction - Customization and Hyperparameter Search
:label:`sec_textprediction_customization`

This tutorial teaches you how to control the hyperparameter tuning process in `TextPredictor` by specifying:

- A custom search space of candidate hyperparameter values to consider.
- Which hyperparameter optimization algorithm should be used to actually search through this space.


```{.python .input}
import numpy as np
import warnings
import autogluon as ag
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Example Data: Stanford Sentiment Treebank

To demonstrate how to customize the configuration in `TextPredictor` and conduct HPO, we will use the Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)) dataset. To reduce the training time, we will subsample 1000 samples for training.


```{.python .input}
from autogluon.core.utils.loaders.load_pd import load
train_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
dev_data = load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
rand_idx = np.random.permutation(np.arange(len(train_data)))[:1000]
train_data = train_data.iloc[rand_idx]
train_data.head(10)
```

## Configuration in TextPredictor

### Pre-configured Hyperparameters in TextPredictor

We provided a series of pre-configured hyperparameters. You may list the keys from `ag_text_presets` via `list_presets`.


```{.python .input}
from autogluon.text import ag_text_presets, list_presets
list_presets()
```

There are two kinds of presets. The `simple_presets` are pre-defined configurations managed by AutoGluon team. We pre-selected the appropriate model 
configurations for different scenarios like `medium_quality_faster_train` or `lower_quality_fast_train`. We also list all the additional presets for advanced users. 
These pre-configured models use different backbones such as ELECTRA, RoBERTa, Multilingual BERT, and different fusion strategies. For example, `electra_small_fuse_late` means to use the ELECTRA-small model as the text backbone and use the late fusion strategy described in ":label:`sec_textprediction_architecture`". By default, we are using `default`, which is the same as `electra_base_fuse_late`. Next, let's try to specify the `presets` in `.fit()` to be `electra_small_fuse_late` and train a model on SSTs.


```{.python .input}
from autogluon.text import TextPredictor
predictor = TextPredictor(path='ag_text_sst_electra_small', eval_metric='acc', label='label')
predictor.set_verbosity(0)
predictor.fit(train_data, presets='electra_small_fuse_late', time_limit=60, seed=123)
```

Here, we try to report the performance of both `f1` and `acc`. However, if you really want to obtain the best F1 score, you should better set 
`eval_metric='f1'` when constructing the predictor.

```{.python .input}
predictor.evaluate(dev_data, metrics=['f1', 'acc'])
```

To visualize the pre-registered hyperparameters, you can call `ag_text_presets.create(key_name)`, e.g.,


```{.python .input}
import pprint
pprint.pprint(ag_text_presets.create('electra_small_fuse_late'))
```

Another way to specify customized config is to directly specify the `hyperparameters` argument in `predict.fit()`. Following is an example


```{.python .input}
predictor.fit(train_data, hyperparameters=ag_text_presets.create('electra_small_fuse_late'),
              time_limit=30, seed=123)
```

### Change Hyperparameter

The pre-registered configurations provide a bunch of good default hyperparameters. 
A common workflow is to first train a model with one of the presets and then tune part of hyperparameters to see if the performance can be better. The following is an example about how to do this in AutoGluon Text. 
You can directly add/changes keys in the hyperparameter dictionary. 
In the example, we change the number of training epochs to 5 and the learning rate to 5E-5.


```{.python .input}
hyperparameters = ag_text_presets.create('electra_small_fuse_late')
hyperparameters['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = 5
hyperparameters['models']['MultimodalTextModel']['search_space']['optimization.lr'] = ag.core.space.Categorical(5E-5)
predictor.fit(train_data, hyperparameters=hyperparameters, time_limit=30, seed=123)
```

### Register Your Own Configuration

You can also register the hyperparameters to `ag_text_presets`. In the following example, 
the `electra_small_fuse_late_train5` preset will use ELECTRA-small as the backbone, 
and will be trained for 5 epochs with weight-decay set to 1E-2. 


```{.python .input}
@ag_text_presets.register()
def electra_small_fuse_late_train5():
    hyperparameters = ag_text_presets.create('electra_small_fuse_late')
    hyperparameters['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = 5
    hyperparameters['models']['MultimodalTextModel']['search_space']['optimization.wd'] = 1E-2
    return hyperparameters

predictor.fit(train_data, presets='electra_small_fuse_late_train5', time_limit=60, seed=123)
```

## Perform HPO over a Customized Search Space with Random Search

To control which hyperparameter values are considered during `fit()`, we specify the `hyperparameters` argument.
Rather than specifying a particular fixed value for a hyperparameter, we can specify a space of values to search over via `ag.space`.
We can also specify which HPO algorithm to use for the search via `search_strategy` (a simple [random search](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) is specified below).
In this example, we search for good values of the following hyperparameters:

- warmup
- number of mid units in the final mlp layer that maps aggregated features to output
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

    hparams['tune_kwargs']['search_strategy'] = 'random'
    return hparams
```

We can now call `fit()` with hyperparameter-tuning over our custom search space. 
Below `num_trials` controls the maximal number of different hyperparameter configurations for which AutoGluon will train models (4 models are trained under different hyperparameter configurations in this case). To achieve good performance in your applications, you should use larger values of `num_trials`, which may identify superior hyperparameter values but will require longer runtimes.


```{.python .input}
predictor_sst_rs = TextPredictor(path='ag_text_sst_random_search', label='label', eval_metric='acc')
predictor_sst_rs.set_verbosity(0)
predictor_sst_rs.fit(train_data,
                      hyperparameters=electra_small_basic_demo_hpo(),
                      time_limit=60 * 2,
                      num_trials=4,
                      seed=123)
```

We can again evaluate our model's performance on separate test data.


```{.python .input}
dev_score = predictor_sst_rs.evaluate(dev_data, metrics=['acc', 'f1'])
print('Best Config = {}'.format(predictor_sst_rs.results['best_config']))
print('Total Time = {}s'.format(predictor_sst_rs.results['total_time']))
print('Accuracy = {:.2f}%'.format(dev_score['acc'] * 100))
print('F1 = {:.2f}%'.format(dev_score['f1'] * 100))
```

## Use Bayesian Optimization + Hyperband

Alternatively, we can use more advanced searchers like the combination of [Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/) and [Hyperband algorithm](https://arxiv.org/pdf/1603.06560.pdf) for HPO.
Hyperband will try multiple hyperparameter configurations simultaneously and will early stop training under poor configurations to free compute resources for exploring new hyperparameter configurations. 
It may be able to identify good hyperparameter values more quickly than other search strategies in your applications. You may refer to [Hyperband and Bayesian Optimization](https://arxiv.org/abs/2003.10865) for more details.


```{.python .input}
hyperparameters = electra_small_basic_demo_hpo()
hyperparameters['tune_kwargs']['search_strategy'] = 'bayesopt_hyperband'
hyperparameters['tune_kwargs']['scheduler_options'] = {'max_t': 15} # Maximal number of epochs for training the neural network
predictor_sst_hb = TextPredictor(path='ag_text_sst_hb', label='label', eval_metric='acc')
predictor_sst_hb.set_verbosity(0)
predictor_sst_hb.fit(train_data,
                     hyperparameters=hyperparameters,
                     time_limit=60 * 2,
                     num_trials=8,
                     seed=123)
```


```{.python .input}
dev_score = predictor_sst_hb.evaluate(dev_data, metrics=['acc', 'f1'])
print('Best Config = {}'.format(predictor_sst_hb.results['best_config']))
print('Total Time = {}s'.format(predictor_sst_hb.results['total_time']))
print('Accuracy = {:.2f}%'.format(dev_score['acc'] * 100))
print('F1 = {:.2f}%'.format(dev_score['f1'] * 100))
```
