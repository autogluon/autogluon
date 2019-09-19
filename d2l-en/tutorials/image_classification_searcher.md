# Image Classification - Configure Your Searcher
:label:`sec_imgsearch`


Beyond simply specifying the space of hyperparameter configurations to search over, you can also tell AutoGluon what strategy it should employ to actually search through this space.  This process of finding good hyperparameters from a given search space is commonly referred to as *hyperparameter optimization* (HPO) or *hyperparameter tuning*.  This tutorial dives into how to configure and create your own customized hyperparameter searcher and use it with `task.fit()`.

We again begin by informing AutoGluon that `image_classification` is the task of interest, and  use the same subset of the `Shopee-IET` dataset as before (recall that as we only specify the `train_path`, a 90/10 train/validation split is automatically performed).  To ensure this tutorial runs quickly, we heavily constrain the allowed time-limits and number of training epochs.

```{.python .input}
from autogluon import image_classification as task

import logging
logging.basicConfig(level=logging.INFO)

dataset = task.Dataset(name='shopeeiet', train_path='~/data/train')

time_limits = 2*60
num_training_epochs = 10
```

Recall that our goal in hyperparameter search is to identify the hyperparameter configuration under which the resulting trained model exhibits the best predictive performance on the validation data (ie. *validation accuracy* for our classification task).  AutoGluon employs a [`Searcher`](../api/autogluon.searcher.html) object that controls which hyperparameter-values from the search space should be explored in the next trial (ie. training run). Certain search procedures such as *Bayesian optimization* may base this choice on all the previous trials that have been executed, using observations of how well the previously tested hyperparameter configurations performed to inform which new hyperparameter configuration seems most promising to try next.  [autogluon.searcher](../api/autogluon.searcher.html) supports various search algorithms for both hyperparameter optimization and architecture search. 

## Random hyperparameter search

The default searcher employed by AutoGluon is random search, which simply tries out new hyperparameter configurations drawn at random from the hyperparameter search space under consideration. You can specify that `task.fit` should employ random hyperparameter search simply by passing the string argument `searcher='random'` (although unnecessary as it is already the default option).

```{.python .input}
results = task.fit(dataset,
                   searcher='random',
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)
```

The resulting validation and test top-1 accuracy obtained through random search within the given `time_limits` and `num_training_epochs` constraints are:

```{.python .input}
print('Top-1 val acc: %.3f' % results.metric)
test_dataset = task.Dataset(name='shopeeiet', test_path='~/data/test')
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

## Hyperparameter search via Bayesian optimization

Instead of random search, AutoGluon can alternatively utilize the more sophisticated strategy of [Bayesian Optimization](../api/autogluon.searcher.html) to identify good hyperparameters.  Bayesian Optimization fits a probabilistic *surrogate model* to estimate the function that relates each hyperparameter configuration to the resulting performance of a model trained under this hyperparameter configuration. This surrogate model can then be used to infer which hyperparameter configurations could plausibly lead to the best predictive performance (ie. those hyperparameter-values that are similar to the top-performing configurations tried so far, as well as the hyperparameter-values that are very dissimilar to all configurations tried so far, since the surrogate model is highly uncertain about their corresponding performance). Within the same time-limit constraints, Bayesian optimization can often produce a superior model  compared to random search by making smarter decisions about which hyperparameter configuration to explore next. Although updating the surrogate model takes time, these updates are generally negligible compared with the neural network training time required to execute a single trial. 

You can specify `task.fit` should find hyperparameters via Bayesian optimization simply by passing the string argument `searcher='bayesopt'`:

```python
results = task.fit(dataset,
                   searcher='bayesopt',
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)
```

The resulting validation and test top-1 accuracy obtained through Bayesian optimization (within the given `time_limits` and `num_training_epochs` constraints) are:

```python
print('Top-1 val acc: %.3f' % results.metric)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

## Customizing the Bayesian optimization based searcher

For those of you familiar with Bayesian optimization, AutoGluon allows you to control many aspects of the Bayesian optimization hyperparameter search process.  For instance, you can specify what kind of surrogate model to use (Gaussian Process, Random Forest, etc), as well as which acquisition function to employ (eg. Expected Improvement, Lower Confidence Bound, etc).  Below, we tell `fit` to perform Bayesian optimization using a Random Forest surrogate model with acquisitions based on Expected Improvement.

```python
results = task.fit(dataset,
                   searcher='bayesopt', 
                   searcher_options={'base_estimator': 'RF', 'acq_func': 'EI'},
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)

print('Top-1 val acc: %.3f' % results.metric)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

Under the hood, Bayesian optimization in AutoGluon is implemented via the [**scikit-optimize**](https://scikit-optimize.github.io/) library, which allows the user to specify all sorts of Bayesian optimization variants. The full functionality of this library is available to use with `task.fit()`, simply by passing the appropriate `kwargs` as `searcher_options`.  Please see the [skopt.optimizer.Optimizer](http://scikit-optimize.github.io/optimizer/index.html#skopt.optimizer.Optimizer) documentation for the full list of keyword arguments that can be passed as `searcher_options` when `searcher='bayesopt'`.

## Create your own searcher

We can also create our own searcher entirely from scratch. Below is an example of a new Searcher class that implements the random search strategy. The two functions you must specify are `get_config` which the AutoGluon training-run scheduler (repeatedly) uses to ask the searcher for a new hyperparameter configuration to use for the next trial, and `update` which updates the internal state maintained by the search algorithm with the performance results obtained by various hyperparameter configurations as soon as their trials have been completed.

```python
from autogluon.searcher import BaseSearcher
class MyRandomSampling(BaseSearcher):
    """Random sampling Searcher for ConfigSpace
    """
    def __init__(self, configspace, dummy_argument=0):
        super(MyRandomSampling, self).__init__(configspace)
        self.dummy_argument = dummy_argument # just used to demonstrate how to pass keyword arguments to your searcher
        self.mystate = {} # used to save and update state for your own algorithm
        
    def get_config(self):
        """Function to sample a new configuration
        """
        # get configuration space
        cs = self.configspace
        # sample the configuration using random sampling
        myconfig = cs.sample_configuration()
        # return the dictionary format
        return myconfig.get_dictionary()

    def update(self, config, reward, mystate):
        """Update the searcher with the newest metric report
        """
        # we automatically track the history results in self._results
        super(MyRandomSampling, self).update(config, reward)
        # you may save and update state for your own algorithm
        self.mystate[config] = mystate
```

Once again, any special keyword arguments that your own searcher may depend on can be specified via `searcher_options`.
We can then use this new searcher inside AutoGluon `fit`:

```python
results = task.fit(dataset,
                   searcher=MyRandomSampling,
                   searcher_options={'dummy_argument': 1},
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)

print('Top-1 val acc: %.3f' % results.metric)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

To understand other aspects of the `fit` function that may be customized, please refer to the [fit API](../api/autogluon.task.image_classification.html#autogluon.task.image_classification.ImageClassification.fit).
