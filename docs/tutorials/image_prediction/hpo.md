# Image Prediction - Search Space and Hyperparameter Optimization (HPO)
:label:`sec_imgadvanced`

While the :ref:`sec_imgquick` introduced basic usage of AutoGluon `fit`, `evaluate`, `predict` with default configurations, this tutorial dives into the various options that you can specify for more advanced control over the fitting process.

These options include:
- Defining the search space of various hyperparameter values for the training of neural networks
- Specifying how to search through your choosen hyperparameter space
- Specifying how to schedule jobs to train a network under a particular hyperparameter configuration.

The advanced functionalities of AutoGluon enable you to use your external knowledge about your particular prediction problem and computing resources to guide the training process. If properly used, you may be able to achieve superior performance within less training time.

**Tip**: If you are new to AutoGluon, review :ref:`sec_imgquick` to learn the basics of the AutoGluon API.

Since our task is to classify images, we will use AutoGluon to produce an [ImagePredictor](/api/autogluon.task.html#autogluon.vision.ImagePredictor):

```{.python .input}
import autogluon.core as ag
from autogluon.vision import ImagePredictor
```

## Create AutoGluon Dataset

Let's first create the dataset using the same subset of the `Shopee-IET` dataset as the :ref:`sec_imgquick` tutorial.
Recall that there's no validation split in original data, a 90/10 train/validation split is automatically performed when `fit` with `train_data`.

```{.python .input}
train_data, _, test_data = ImagePredictor.Dataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
```

## Specify which Networks to Try

We start with specifying the pretrained neural network candidates.
Given such a list, AutoGluon tries to train different networks from this list to identify the best-performing candidate.
This is an example of a :class:`autogluon.core.space.Categorical` search space, in which there are a limited number of values to choose from.

```{.python .input}
model = ag.Categorical('resnet18_v1b', 'mobilenetv3_small')

# you may choose more than 70+ available model in the model zoo provided by GluonCV:
model_list = ImagePredictor.list_models()
```

## Specify the training hyper-parameters

Similarly, we can manually specify many crucial hyper-parameters, with specific value or search space(`autogluon.core.space`).


```{.python .input}
batch_size = 8
lr = ag.Categorical(1e-2, 1e-3)
```

## Search Algorithms

In AutoGluon, `autogluon.core.searcher` supports different search search strategies for both hyperparameter optimization and architecture search.
Beyond simply specifying the space of hyperparameter configurations to search over, you can also tell AutoGluon what strategy it should employ to actually search through this space.
This process of finding good hyperparameters from a given search space is commonly referred to as *hyperparameter optimization* (HPO) or *hyperparameter tuning*.
`autogluon.core.scheduler` orchestrates how individual training jobs are scheduled.
We currently support FIFO (standard) and Hyperband scheduling, along with search
by random sampling or Bayesian optimization. These basic techniques are rendered
surprisingly powerful by AutoGluon's support of asynchronous parallel execution.

### Bayesian Optimization

Here is an example of using Bayesian Optimization using :class:`autogluon.core.searcher.GPFIFOSearcher`.

Bayesian Optimization fits a probabilistic *surrogate model* to estimate the
function that relates each hyperparameter configuration to the resulting performance
of a model trained under this hyperparameter configuration. Our implementation makes
use of a Gaussian process surrogate model along with expected improvement as
acquisition function. It has been developed specifically to support asynchronous
parallel evaluations.

```{.python .input}
hyperparameters={'model': model, 'batch_size': batch_size, 'lr': lr, 'epochs': 2}
predictor = ImagePredictor()
predictor.fit(train_data, search_strategy='bayesopt', num_trials=2, time_limit=60*10, hyperparameters=hyperparameters)
print('Top-1 val acc: %.3f' % predictor.fit_summary()['valid_acc'])
```

The BO searcher can be configured by `search_options`, see
:class:`autogluon.core.searcher.GPFIFOSearcher`. Load the test dataset and evaluate:

```{.python .input}
top1, top5 = predictor.evaluate(test_data)
print('Test acc on hold-out data:', top1)
```

Note that `num_trials=2` above is only used to speed up the tutorial. In normal
practice, it is common to only use `time_limits` and drop `num_trials`.

### Hyperband Early Stopping

AutoGluon currently supports scheduling trials in serial order and with early
stopping (e.g., if the performance of the model early within training already
looks bad, the trial may be terminated early to free up resources).
Here is an example of using an early stopping scheduler
:class:`autogluon.core.scheduler.HyperbandScheduler`. `scheduler_options` is used
to configure the scheduler. In this example, we run Hyperband with a single
bracket, and stop/go decisions are made after 1 and 2 epochs (`grace_period`,
`grace_period * reduction_factor`):

```{.python .input}
hyperparameters.update({
  'search_strategy': 'hyperband',
  'grace_period': 1
  })
```

The `fit`, `evaluate` and `predict` processes are exactly the same, so we will skip training to save some time.

### Bayesian Optimization and Hyperband ###

While Hyperband scheduling is normally driven by a random searcher, AutoGluon
also provides Hyperband together with Bayesian optimization. The tuning of expensive
DL models typically works best with this combination.

```{.python .input}
hyperparameters.update({
  'search_strategy': 'bayesopt_hyperband',
  'grace_period': 1
  })
```

For a comparison of different search algorithms and scheduling strategies, see :ref:`course_alg`.
For more options using `fit`, see :class:`autogluon.vision.ImagePredictor`.
