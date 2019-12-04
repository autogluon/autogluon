import math
import autogluon as ag
from autogluon import ImageClassification as task
from autogluon import config_choice

# root = /your_path/sy_datasets/
root = '/home/ubuntu/workspace/1107gluoncv_cla/data/'

# dataset = 'dogs-vs-cats-redux-kernels-edition/'
dataset = 'aerial-cactus-identification/'
# dataset = 'plant-seedlings-classification/'
# dataset = 'fisheries_Monitoring/'
# dataset = 'dog-breed-identification/'

target = config_choice(dataset, root)

classifier = task.fit(dataset = task.Dataset(target['dataset']),
                      classes = target['classes'],
                      net = target['net'],
                      optimizer = target['optimizer'],
                      lr_scheduler = target['lr_scheduler'],
                      epochs = target['epochs'],
                      ngpus_per_trial = target['ngpus_per_trial'],
                      num_trials = target['num_trials'],
                      batch_size = target['batch_size'],
                      verbose=True,
                      plot_results=True,
                      tricks = target['tricks'],
                      lr_config = target['lr_config'])# ->
print(classifier)

ag.done()
