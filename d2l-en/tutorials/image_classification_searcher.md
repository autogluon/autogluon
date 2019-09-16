# Image Classification - Configure Your Searcher
:label:`sec_imgsearch`

This tutorial dives into how to configure and create your own searcher and use the customized searcher in the fitting process.

We again begin by letting AutoGluon know that `image_classification` is the task of interest: 

```{.python .input}
from autogluon import image_classification as task

import logging
logging.basicConfig(level=logging.INFO)
```

## Use random searcher

Let's first create the dataset using the same subset of the `Shopee-IET` dataset as before.
Recall that as we only specify the `train_path`, a 90/10 train/validation split is automatically performed.
We also set the time limits and training epochs to ensure the demo run quickly.

[autogluon.searcher](../api/autogluon.searcher.html)
will support both basic and advanced search algorithms for both hyperparameter optimization and architecture search. 
The default searcher is random search. The easiest way to specify random search is via the string name `searcher='random'`.

```{.python .input}
dataset = task.Dataset(name='shopeeiet', train_path='~/data/train')

time_limits = 2*60
num_training_epochs = 10

results = task.fit(dataset,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)
```

The validation and test top-1 accuracy are:

```{.python .input}
print('Top-1 val acc: %.3f' % results.metric)
test_dataset = task.Dataset(name='shopeeiet', test_path='~/data/test')
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

## Use Bayesian Optimization based searcher

We could easily use [Bayesian Optimization](../api/autogluon.searcher.html) as searcher to optimize the hyperparameter selection. Bayesian Optimization tries to fit a probailistic model to the function evaluations.
Based on our experiments on CIFAR10, Bayesian Optimization could obtain over 0.22 improvements compared to random search.
We could simply specify Bayesian Optimization via string name and use it in the `fit` function:

```{.python .input}
searcher = 'bayesopt'

results = task.fit(dataset,
                   searcher=searcher,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)
```

The validation and test top-1 accuracy are:

```{.python .input}
print('Top-1 val acc: %.3f' % results.metric)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

## Configure Bayesian Optimization based searcher

We could specify the variables in the Bayesian Optimization to potentially get better optimization control over the fit procedure.

```{.python .input}
#TODO: add bayesopt config example jonas
```

## Create your own searcher

We could also create our own searcher. An example of creating the customized searcher could be found:

```{.python .input}
#TODO: jonas add more description about each code block
from autogluon.searcher import BaseSearcher
class MyRandomSampling(BaseSearcher):
    """Random sampling Searcher for ConfigSpace
    """
    def __init__(self, configspace):
        super(MyRandomSampling, self).__init__(configspace)
        # you may save and update state for your own algorithm
        self.mystate = {}
        
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

We then could use it in AutoGluon `fit`.

```{.python .input}
results = task.fit(dataset,
                   searcher=MyRandomSampling,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)
```

The validation and test top-1 accuracy are:

```{.python .input}
print('Top-1 val acc: %.3f' % results.metric)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

For more complete usage of `fit` function, please refer to the [fit API](../api/autogluon.task.image_classification.html#autogluon.task.image_classification.ImageClassification.fit).
