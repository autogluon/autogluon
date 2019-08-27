# Image Classification - Advanced

While the [beginner tutorial](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/build/task/image_classification_beginner.html) introduced basic usage of AutoGluon `fit` and `predict` with default options, this tutorial dives into the various options that you can specify for more advanced control over the model-building process.

These options include: defining the search space of model hyperparameter values to consider, specifying how to actually search through this hyperparameter space, and how to schedule the jobs which actually train a network under a particular hyperparameter configuration.
The advanced functionalities of AutoGluon allow you to leverage your external knowledge about your particular prediction problem and computing infrastructure to guide the model-building process. If properly utilized, you may be able to achieve superior performance within less training time.


We again begin by letting AutoGluon know that `image_classification` is the task of interest: 

```python
from autogluon import image_classification as task

import logging
logging.basicConfig(level=logging.INFO)
```

## Create AutoGluon Dataset

Let's first create the dataset using the same subset of the `Shopee-IET` dataset as before.
Recall that as we only specify the `train_path`, a 90/10 train/validation split is automatically performed.

```python
dataset = task.Dataset(name='shopeeiet', train_path='data/train')
```

## Understanding default settings of AutoGluon's fit()

To ensure this demo runs quickly, we expect each call to `fit` can be finished within 5min,
and individual training runs (also referred to as "trials") each last for 10 epochs.

```python
time_limits = 5*60
num_training_epochs = 10
```

We first again use the default arguments of the `fit` function to train a model:

```python
results = task.fit(dataset,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)
```

The validation and test top-1 accuracy are:

```python
print('Top-1 val acc: %.3f' % results.metric)
test_dataset = task.Dataset(name='shopeeiet', test_path='data/test')
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

Let's now dive deeper into the default settings of `fit` that an advanced user may wish to alter.

Recall that rather than training an image classification neural network from scratch, AutoGluon by default first loads a network that has already been pretrained on another image dataset and then continues training this network on your provided dataset (after appropriately modifying the output layer for the current task).  Let's inspect which pretrained neural network candidates AutoGluon tries to build off of by default for image classification tasks:

```python
print('Default models:')
print(results.metadata['nets'])
```

We can also look up the default optimizers used by `fit` to train each neural network (ie. to update the weight parameters based on mini-batches of training data):
 
```python
print('Default optimizers:')
print(results.metadata['optimizers'])
```

Beyond which pretrained model and which optimizer to use, deep learning in general involves tons of other design choices, which we collectively refer to as "hyperparameters". Given possible values of the hyperparameters to try out, we require a smart strategy to efficiently find those hyperparameter values which will produce the best classifier. Strategies might include techniques such as grid/random-search, hyperband, Bayesian optimization, etc. In AutoGluon, which hyperparameter-search strategy to use is specified by a `Searcher` object.
The default searcher used in `fit` for image classification is:

```python
print('Default searcher:')
print(results.metadata['searcher'])
```

When the Searcher returns a particular hyperparameter configuration to try out, we must train a neural network under the specified configuration settings, a process referred to as a "trial". In parallel/distributed settings, we may wish to run multiple trials simultaneously in order to try out more hyperparameter configurations in less time. In AutoGluon, how trials are orchestrated is controlled by a `Scheduler` object.
The default trial scheduler used in `fit` for image classification is:

```python
print('Default trial scheduler:')
print(results.metadata['trial_scheduler'])
```

When we have already trained many networks under many hyperparameter configurations and see that a current trial is performing very poorly in comparison, it may be wise to infer this is not the best hyperparameter configuration and to simply terminate the trial right away. This way, these compute resources are freed up to explore more promising hyperparameter configurations.  

We can retrieve AutoGluon's default stop-criterion and resources used for each trial using:

```python
print('Used stop criterion:')
print(results.metadata['stop_criterion'])

print('Used resources for each trial:')
print(results.metadata['resources_per_trial'])
```

TODO: add code snippet showing how to specify max_gpu max_cpu per trial.


In general, we can expect fairly strong predictive performance using the default settings in `fit`, if given enough computing resources and runtime.
In order to achieve superior performance with limited compute, we can manually specify different settings in `fit` based on prior knowledge or recent research papers.
One important aspect of this involves defining the space of hyperparameter values to search over.
We will use `autogluon.Nets` and `autogluon.Optimizers` as
examples to show how to specify a custom hyperparameter search space, and ensure it is used by the `fit` function.



## Specify which pretrained networks to try

We start with specifying the pretrained neural network candidates.
In AutoGluon, the network candidates are represented as [autogluon.Nets](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.network.Nets),
which is simply a list of networks.  
Given such a list, AutoGluon will try training different networks from this list to identify the best-performing candidate.
This is an example of a "categorical" search space, in which there are a limited number of values to choose from.

For more information regarding categorical search spaces, please
refer to the [search space API](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.space.List).

In addition to the default network candidates, let's also add `resnet50` to the search space:

```python
import autogluon as ag
net_list = ['resnet18_v1',
            'resnet34_v1',
            'resnet50_v1']

# default net list for image classification would be overwritten
# if net_list is provided
nets = ag.Nets(net_list)

print(nets)
```

Let's then call `fit` using these manually-specified network candidates, and evaluate on validation and test data.

```python
results = task.fit(dataset,
                   nets,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)
```

The validation and test top-1 accuracy are:

```python
print('Top-1 val acc: %.3f' % results.metric)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

## Specify which optimizers to try

Similarly, we can manually specify which of optimizer candidates to try, in order to further improve the results.
In AutoGluon, [autogluon.Optimizers](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.optim.Optimizers) 
defines a list of optimization algorithms, from which we can construct another search space to identify which optimizer works best for our task (as well as what are the best hyperparameter settings for this optimizer).

Like `autogluon.Nets`, the choice of which optimizer to use again corresponds to a [categorical](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.space.List) search space:

```python
optimizers_default = ag.Optimizers(['sgd', 'adam'])
```

Additionally, we can customize the optimizer-specific hyperparameters as another search space.
As an example for both `Adam` and `SGD`, we can configure the learning rate and weight decay in a continuous-valued search space.
Below, we specify that this space should be searched on a log-linear scale, providing two numbers to lower and upper bound the space of values to try.  
Additionally, the momentum in `SGD` is configured as another continuous search space (searched on a linear scale), where the
two numbers are also the lower and upper bounds of the space.


```python
adam_opt = ag.optims.Adam(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
                          wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
sgd_opt = ag.optims.SGD(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
                        momentum=ag.space.Linear('momentum', 0.85, 0.95),
                        wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
optimizers = ag.Optimizers([adam_opt, sgd_opt])

print(optimizers)
```

Please refer to [log search space](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.space.Log) and [linear search space](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.space.Linear) for more details.

We then put the new network and optimizer search space together in the call to `fit` and might expect better results if we have made smart choices:

```python
results = task.fit(dataset,
                   nets,
                   optimizers,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)
```

The validation and test top-1 accuracy are:

```python
print('Top-1 val acc: %.3f' % results.metric)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

## Specify a hyperparameter search strategy and how to schedule trials

[autogluon.searcher](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/backend.html#autogluon-searcher)
will support both basic and advanced search algorithms for
both hyperparameter optimization and architecture search.  
Advanced search algorithms, such as Bayesian Optimization, Population-Based Training, and BOHB, are coming soon.  
We currently support Hyperband and random search. Although these are simple techinques, they can be surprisingly powerful when parallelized, which easily enabled in AutoGluon.
The easiest way to specify random search is via the string name:

```python
searcher = 'random'
```

In AutoGluon, [autogluon.scheduler](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/backend.html#autogluon-scheduler) orchestrates how individual training runs are scheduled. Separating the logic of the individual training code that constitutes a trial from the scheduling of trials is most useful in parallel/distributed settings.

AutoGluon currently supports scheduling trials in serial order and with early stopping (eg. if the performance of the model early within training already looks bad, the trial may be terminated early to free up resources).
We support a serial [FIFO scheduler](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/backend.html#autogluon-scheduler.FIFO_Scheduler)
and an early stopping scheduler: [Hyperband](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/backend.html#autogluon-scheduler.Hyperband_Scheduler).
Which scheduler to use is easily specified via the string name:

```python
trial_scheduler_fifo = 'fifo'
trial_scheduler = 'hyperband'
```

Let's call `fit` with the Searcher and Scheduler specified above, 
and evaluate the resulting model on both validation and test datasets:

```python
results = task.fit(dataset,
                   nets,
                   optimizers,
                   searcher=searcher,
                   trial_scheduler=trial_scheduler,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)
```

The validation and test top-1 accuracy are:

```python
print('Top-1 val acc: %.3f' % results.metric)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```


Let's now use the same image as used in our [beginner tutorial](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/build/task/image_classification_beginner.html) to generate a predicted label and the corresponding confidence.
Note that even though we called `fit` with non-default settings, we can still use the `predict` function just as before:

```python
image = './data/test/BabyShirt/BabyShirt_323.jpg'
ind, prob = task.predict(image)
print('The input picture is classified as [%s], with probability %.2f.' %
      (dataset.train.synsets[ind.asscalar()], prob.asscalar()))
```


Beyond what we described here, there are many other aspects of `fit` that an advanced user can control.  
Please refer to the [fit API](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.task.image_classification.ImageClassification.fit).
