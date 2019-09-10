"""3. Image Classification - Advanced
============================================

In the [first tutorial](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/build/task/image_classification_beginner.html), we introduce the basic usage of AutoGluon.
In this tutorial, we want to illustrate the advanced usage of AutoGluon, such as configuring
the search space of the hyper-parameters, searchers and trial schedulers.
By leveraging the advanced functionalies of AutoGluon, we should be able to achieve better
image classification results.

We would first import autogluon.
"""
from autogluon import image_classification as task

import logging
logging.basicConfig(level=logging.INFO)

################################################################
# Create AutoGluon Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Let's first create the dataset using the same sampled `Shopee-IET` dataset.
# We only specify the `train_path`, then the train/validation split based on 9/1 would
# be automatically performed.

dataset = task.Dataset(name='shopeeiet', train_path='data/train')

################################################################
# Create AutoGluon Fit Using Default Configurations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For demo purpose, we expect each `fit` procedure can be finished within 5min,
# and each trials last for 10 epochs.

#TODO
time_limits = 30
num_training_epochs = 10

################################################################
# We then use the default configurations of `fit` function
# to produce results based on the train data.

results = task.fit(dataset,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)

################################################################
# The validation and test top-1 accuracy are:

print('Top-1 val acc: %.3f' % results.metric)
test_dataset = task.Dataset(name='shopeeiet', test_path='data/test')
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)

################################################################
# Let's dive a little bit deeper to take a look at the default
# configuration of the `fit`.
# First, we could obtain the default model candidates
# for the image classification task.

print('Default models:')
print(results.metadata['nets'])

################################################################
# We can also look up the default optimizers used in the `fit`.

print('Default optimizers:')
print(results.metadata['optimizers'])

################################################################
# The default searcher used in the `fit` is:

print('Default searcher:')
print(results.metadata['searcher'])

################################################################
# The default trial scheduler used in the `fit` is:

print('Default trial scheduler:')
print(results.metadata['trial_scheduler'])

################################################################
# We could also retrieve the used stop criterion and resources used for each trial using:

print('Used stop criterion:')
print(results.metadata['stop_criterion'])

print('Used resources for each trial:')
print(results.metadata['resources_per_trial'])

################################################################
# For more details of `fit`,
# please refer to the [fit API](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.task.image_classification.ImageClassification.fit).


################################################################
# We could expect fair results using the default configurations in `fit`.
# However, in order to achieve better image classification results,
# we could configure the search space based on prior knowledge or existing literature.
# We will use `autogluon.Nets` and `autogluon.Optimizers` as
# examples to show how to configure the corresponding search space, as well as
# how to put the configured search space into the `fit` function.


################################################################
# Create AutoGluon Fit with AutoGluon Nets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We start with configuring the network candidates.
# In AutoGluon, the network candidates are represented as [autogluon.Nets](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.network.Nets),
# which is a list of networks, and allows search for the
# best network from a list of provided (or default) networks by choosing
# the best architecture regarding to each network.
# The search space is categorical. For more information regarding to the categorical search space, please
# refer to the [search space API](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.space.List).
# Compare to default network candidates, we add `resnet50` to the search space.

import autogluon as ag
net_list = ['resnet18_v1',
            'resnet34_v1',
            'resnet50_v1']

# default net list for image classification would be overwritten
# if net_list is provided
nets = ag.Nets(net_list)

print(nets)

################################################################
# Let's then create `fit` using the configured network candidates, and evaluate on validation and test data.

results = task.fit(dataset,
                   nets,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)

################################################################
# The validation and test top-1 accuracy are:

print('Top-1 val acc: %.3f' % results.metric)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)


################################################################
# Create AutoGluon Fit with AutoGluon Optimizers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Similarly, we could additionally configure the hyper-parameters of optimizer candidates.\
# to further improve the results.
# In AutoGluon, [autogluon.Optimizers](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.optim.Optimizers),
# defines a list of optimization algorithms that
# allows search for the best optimization algorithm from a list of provided
# (or default) optimizers by choosing the best hyper-parameters regarding to each
# optimizer.

################################################################
# Similar to network search space,
# we could use the task-specific default optimizer configuration to construct the optimizer search
# space, where the search space is [categorical](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.space.List).

optimizers_default = ag.Optimizers(['sgd', 'adam'])

################################################################
# Besides, we could customize the hyper-parameters of the optimizer in the search space.
# For example, we configure the learning rate and weight decay in log linear search space for both `Adam` and `SGD`,
# where the two numbers are lower and upper bound of the space.
# Please refer to [log search space](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.space.Log) for more information.
# Additionally, the momentum in `SGD` is configured as linear search space, where the
# two numbers are also the lower and upper bounds of the space.
# Please refer to [linear search space](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.space.Linear) for details.

adam_opt = ag.optimizers.Adam(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
                              wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
sgd_opt = ag.optimizers.SGD(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
                            momentum=ag.space.Linear('momentum', 0.85, 0.95),
                            wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
optimizers = ag.Optimizers([adam_opt, sgd_opt])

print(optimizers)

################################################################
# We then put the configured network and optimizer search space together in the `fit` to expect better results.

results = task.fit(dataset,
                   nets,
                   optimizers,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)


################################################################
# The validation and test top-1 accuracy are:

print('Top-1 val acc: %.3f' % results.metric)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)


#################################################################
# Create AutoGluon Fit with Search Algorithm and Trial Scheduler
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# [autogluon.searcher](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/backend.html#autogluon-searcher)
# will support both basic and advanced search algorithms for
# both hyper-parameter optimization and architecture search. We now
# support random search. More search algorithms, such as Bayes Optimization and BOHB, are coming soon.
# The easiest way to use the random search is to specify the string name.

searcher = 'random'

################################################################
# In AutoGluon, [autogluon.scheduler](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/backend.html#autogluon-scheduler)
# supports scheduling trials in serial order and with
# early stopping.
#
# We support [FIFO scheduler](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/backend.html#autogluon-scheduler.FIFO_Scheduler) (in serial order)
# and early stopping scheduler: [Hyperband](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/backend.html#autogluon-scheduler.Hyperband_Scheduler).
# TODO We can simply use string name to specify the schedulers:

trial_scheduler_fifo = 'fifo'
trial_scheduler = 'hyperband'


##################################################################
# Let's then put the above mentioned configurations in the `fit`,
# and test the results on both validation and test datasets.

results = task.fit(dataset,
                   nets,
                   optimizers,
                   searcher=searcher,
                   trial_scheduler=trial_scheduler,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)

################################################################
# The validation and test top-1 accuracy are:

print('Top-1 val acc: %.3f' % results.metric)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)

###############################################################################
# Let's use the same image as used in [first tutorial](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/build/task/image_classification_beginner.html)
# to generate the predicted label and the corresponding probability.

image = './data/test/BabyShirt/BabyShirt_323.jpg'
ind, prob = task.predict(image)
print('The input picture is classified as [%s], with probability %.2f.' %
      (dataset.train.synsets[ind.asscalar()], prob.asscalar()))

################################################################
# For more usage and configurations of the `fit`, please refer to the [fit API](http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/frontend.html#autogluon.task.image_classification.ImageClassification.fit).