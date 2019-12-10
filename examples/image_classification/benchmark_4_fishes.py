import os, glob
import autogluon as ag
from autogluon import ImageClassification as task
from autogluon import config_choice

from gluoncv.utils import makedirs
local_path = os.path.dirname(__file__)

# data
# dataset = 'dogs-vs-cats-redux-kernels-edition/' #ok
# dataset = 'aerial-cactus-identification/' #ok
# dataset = 'plant-seedlings-classification/' #ok
dataset = 'fisheries_Monitoring/'
# dataset = 'dog-breed-identification/'
# dataset = 'shopee-iet-machine-learning-competition/' #ok
# dataset = 'shopee-iet/'

target = config_choice(dataset, local_path)
load_dataset = task.Dataset(target['dataset'])

# fit funcation
# classifier = task.fit(dataset = load_dataset,
#                       net = target['net'],
#                       optimizer = target['optimizer'],
#                       epochs = 1,
#                       ngpus_per_trial =1 ,
#                       num_trials = 2,
#                       batch_size = 64,
#                       verbose=True,
#                       # tricks = tricks,
#                       # lr_config = lr_config,
#                       plot_results = True)

classifier = task.fit(dataset = task.Dataset(target['dataset']),
                      net = target['net'],
                      optimizer = target['optimizer'],
                      epochs = target['epochs'],
                      ngpus_per_trial = target['ngpus_per_trial'],
                      num_trials = target['num_trials'],
                      batch_size = target['batch_size'],
                      verbose=True,
                      # tricks = tricks,
                      # lr_config = lr_config,
                      plot_results = True)

# print('Top-1 val acc: %.3f' % classifier.results['best_reward'])

# save
# ag.utils.save(classifier.state_dict(), "ss")
# classifier.load_state_dict(ag.utils.load("ss"))
# ag.save(scheduler.state_dict(), 'checkpoint.ag')
# scheduler.load_state_dict(ag.load('checkpoint.ag'))

# Valid dataset Evaluate
# test_dataset = task.Dataset(target['dataset'].replace('train', 'val'))
# test_reward = classifier.evaluate(test_dataset)
# print('Top-1 test acc: %.3f' % test_reward)

# fit_summary
makedirs(dataset)
summary = classifier.fit_summary(output_directory=dataset, verbosity = 3)
print(summary)

# predict on Test Dataset ok
# test_dataset = task.Dataset(target['dataset'].replace('train', 'test'))
# inds, probs = classifier.predict(test_dataset)
# print(inds, probs)

# Submit test predictions to Kaggle (Optional)
# generate_csv(inds, path)
csv_path = target['dataset'].replace('train', 'sample_submission.csv')
ag.utils.generate_csv_submission(csv_path, dataset, load_dataset, classifier, local_path)

ag.done()
