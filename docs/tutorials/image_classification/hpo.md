# Image Classification - Search Space and HPO
:label:`sec_imgadvanced`

While the :ref:`sec_imgquick` introduced basic usage of AutoGluon `fit`, `evaluate`, `predict` with default configurations, this tutorial dives into the various options that you can specify for more advanced control over the fitting process.

These options include: defining the search space of various hyperparameter values regarding to the training of neural networks, specifying how to actually search through this hyperparameter space, and how to schedule each particular job which actually train a network under a particular hyperparameter configuration.
The advanced functionalities of AutoGluon allow you to leverage your external knowledge about your particular prediction problem and computing resources to guide the training process. If properly utilized, you may be able to achieve superior performance within less training time.


We again begin by letting AutoGluon know that `ImageClassification` is the task of interest: 

```{.python .input}
import autogluon as ag
from autogluon import ImageClassification as task
```

## Create AutoGluon Dataset

Let's first create the dataset using the same subset of the `Shopee-IET` dataset as before.
Recall that as we only specify the `train_path`, a 90/10 train/validation split is automatically performed.

```{.python .input}
filename = ag.download('http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/data.zip')
ag.unzip(filename)
```

```{.python .input}
dataset = task.Dataset(train_path='data/train')
```

## Understanding default configurations of AutoGluon's fit

To ensure this demo runs quickly, we expect each call to `fit` can be finished within minutes,
and individual training runs (also referred to as `trials`) each last for 10 epochs.

```{.python .input}
time_limits = 1*60
epochs = 2
```

We first again use the default arguments of the `fit` function to train the neural networks:

```{.python .input}
classifier = task.fit(dataset,
                      time_limits=time_limits,
                      epochs=epochs,
                      ngpus_per_trial=1)
```

The validation and test top-1 accuracy are:

```{.python .input}
print('Top-1 val acc: %.3f' % classifier.results[classifier.results['reward_attr']])
test_dataset = task.Dataset(test_path='~/data/test')
test_acc = classifier.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

Let's now dive deeper into the default settings of `fit` that an advanced user may wish to alter.

Recall that rather than training image classification models from scratch, AutoGluon by default first loads networks that has already been pretrained on another image dataset and then continues training the networks on your provided dataset (after appropriately modifying the output layer for the current task). Let's inspect which pretrained neural network candidates are by default for image classification tasks:

```{.python .input}
print('Default Search Info:')
print(classifier.results['metadata'])
```

In general, we can expect fairly strong predictive performance using the default configurations in `fit`, if given enough computing resources and runtime.
In order to achieve superior performance with limited computation, we can manually specify different configurations in `fit` based on prior knowledge or research papers.
One important aspect of this involves defining the space of hyperparameter values to search over.
We will use `autogluon.Nets` and `autogluon.Optimizers` as
examples to show how to specify a custom hyperparameter search space, and ensure it is used by the `fit` function.



## Specify which pretrained networks to try

We start with specifying the pretrained neural network candidates.
Given such a list, AutoGluon will try training different networks from this list to identify the best-performing candidate.
This is an example of a `Categorial` search space, in which there are a limited number of values to choose from.

For more information regarding categorical search spaces, please
refer to the :meth:`autogluon.space.Categorial`.

```{.python .input}
import autogluon as ag
nets = ag.space.Categorical('resnet18_v1', 'resnet34_v1','resnet50_v1')

print(nets)
```

## Specify which optimizers to try

Similarly, we can manually specify which of optimizer candidates to try, in order to further improve the results.
In AutoGluon, [autogluon.Optimizers](../api/autogluon.optimizer.html) 
defines a list of optimization search_strategys, from which we can construct another search space to identify which optimizer works best for our task (as well as what are the best hyperparameter configurations for this optimizer).

Additionally, we can customize the optimizer-specific hyperparameters as another search space.
As an example for both `Adam` and `SGD`, we can configure the learning rate and weight decay in a continuous-valued search space.
Below, we specify that this space should be searched on a log-linear scale, providing two numbers to be lower and upper bounds for the space of values to try.  

Additionally, the momentum in `SGD` is configured as another continuous search space on a linear scale, where the two numbers are also the lower and upper bounds of the space. Moreover, to achieve better results, we could also configure the learning rate scheduler as a categorical space, where in this example, we enable decay and cosine based learning rate schedulers for SGD optimizer.


```{.python .input}
sgd_opt = ag.optimizer.SGD(learning_rate=ag.space.Real(1e-4, 1e-1, log=True),
                           momentum=ag.space.Real(0.85, 0.95),
                           wd=ag.space.Real(1e-6, 1e-2, log=True))
adam_opt = ag.optimizer.Adam(learning_rate=ag.space.Real(1e-4, 1e-1, log=True),
                             wd=ag.space.Real(1e-6, 1e-2, log=True))
optimizers = ag.space.Categorical(sgd_opt, adam_opt)

print(optimizers)
```

Please refer to [log search space](../api/autogluon.space.html#autogluon.space.Log) and [linear search space](../api/autogluon.space.html#autogluon.space.Linear) for more details.

Besides, we could also specify the candidates of learning rate schedulers which are typically leveraged to achieve better results.
We then put the new network and optimizer search space and the learning rate schedulers together in the call to `fit` and might expect better results if we have made smart choices:


## Specify a hyperparameter search strategy and how to schedule trials

In AutoGluon, :meth:`autogluon.searcher` supports different search search_strategys for both hyperparameter optimization and architecture search.
Beyond simply specifying the space of hyperparameter configurations to search over, you can also tell AutoGluon what strategy it should employ to actually search through this space. 
This process of finding good hyperparameters from a given search space is commonly referred to as *hyperparameter optimization* (HPO) or *hyperparameter tuning*. 
:meth:`autogluon.scheduler` orchestrates how individual training jobs are scheduled.
We currently support random search, Hyperband and Bayesian Optimization. Although these are simple techniques, they can be surprisingly powerful when parallelized, which can be easily enabled in AutoGluon.

### Search algorithm

Here we use BayesOpt as an example of advanced search algorithm.
Instead of random search, AutoGluon can alternatively utilize the more sophisticated strategy of :class:`autogluon.searcher.SKoptSearcher` to identify good hyperparameters.  Bayesian Optimization fits a probabilistic *surrogate model* to estimate the function that relates each hyperparameter configuration to the resulting performance of a model trained under this hyperparameter configuration. This surrogate model can then be used to infer which hyperparameter configurations could plausibly lead to the best predictive performance (ie. those hyperparameter-values that are similar to the top-performing configurations tried so far, as well as the hyperparameter-values that are very dissimilar to all configurations tried so far, since the surrogate model is highly uncertain about their corresponding performance). Within the same time-limit constraints, Bayesian optimization can often produce a superior model  compared to random search by making smarter decisions about which hyperparameter configuration to explore next. Although updating the surrogate model takes time, these updates are generally negligible compared with the neural network training time required to execute a single trial. 

For those of you familiar with Bayesian optimization, AutoGluon allows you to control many aspects of the Bayesian optimization hyperparameter search process.  For instance, you can specify what kind of surrogate model to use (Gaussian Process, Random Forest, etc), as well as which acquisition function to employ (eg. Expected Improvement, Lower Confidence Bound, etc).  Below, we tell `fit` to perform Bayesian optimization using a Random Forest surrogate model with acquisitions based on Expected Improvement.

```{.python .input}
classifier = task.fit(dataset,
                   search_strategy='skopt', 
                   search_options={'base_estimator': 'RF', 'acq_func': 'EI'},
                   time_limits=time_limits,
                   epochs=epochs,
                   ngpus_per_trial=1)

print('Top-1 val acc: %.3f' % classifier.results[classifier.results['reward_attr']])
test_acc = classifier.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```

Under the hood, Bayesian optimization in AutoGluon is implemented via the [**scikit-optimize**](https://scikit-optimize.github.io/) library, which allows the user to specify all sorts of Bayesian optimization variants. The full functionality of this library is available to use with `task.fit()`, simply by passing the appropriate `kwargs` as `search_options`.  Please see the [skopt.optimizer.Optimizer](http://scikit-optimize.github.io/optimizer/index.html#skopt.optimizer.Optimizer) documentation for the full list of keyword arguments that can be passed as `search_options` when `search_strategy='skopt'`.


### Early stopping

AutoGluon currently supports scheduling trials in serial order and with early stopping (eg. if the performance of the model early within training already looks bad, the trial may be terminated early to free up resources).
We support a serial :class:`autogluon.scheduler.FIFOScheduler`
and an early stopping scheduler :class:`autogluon.scheduler.HyperbandScheduler`.
Which scheduler to use is easily specified via the string name:

```{.python .input}
search_strategy = 'hyperband'
```

Let's call `fit` with the Searcher and Scheduler specified above,
and evaluate the resulting model on both validation and test datasets:

```{.python .input}
classifier = task.fit(dataset,
                   nets,
                   optimizers,
                   lr_scheduler=ag.space.Categorical('poly', 'cosine'),
                   search_strategy=search_strategy,
                   time_limits=time_limits,
                   epochs=4,
                   ngpus_per_trial=1)
```

The validation and test top-1 accuracy are:

```{.python .input}
print('Top-1 val acc: %.3f' % classifier.results[classifier.results['reward_attr']])
test_acc = classifier.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
```


Let's now use the same image as used in :ref:`sec_imgquick` to generate a predicted label and the corresponding confidence.

```{.python .input}
image = './data/test/BabyShirt/BabyShirt_323.jpg'
ind, prob = classifier.predict(image)
print('The input picture is classified as [%s], with probability %.2f.' %
      (dataset.init().synsets[ind.asscalar()], prob.asscalar()))
```


Beyond what we described here, there are many other aspects of `fit` that an advanced user can control.
For that, please refer to the [fit API](../api/autogluon.task.image_classification.html#autogluon.task.image_classification.ImageClassification.fit).

Finish and exit:
```{.python .input}
ag.done()
```
