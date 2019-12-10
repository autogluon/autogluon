import os, glob
import autogluon as ag
import logging
from autogluon import ImageClassification as task
from autogluon import config_choice
from gluoncv.utils import makedirs

# data
local_path = os.path.dirname(__file__)
# data_path = '/your_path/'
data_path = '/home/ubuntu/workspace/data/dataset/'
# dataset = 'dogs-vs-cats-redux-kernels-edition/' #sub ok
# dataset = 'aerial-cactus-identification/' # ok
dataset = 'plant-seedlings-classification/' # sub ok
# dataset = 'fisheries_Monitoring/' # sub ok
# dataset = 'dog-breed-identification/'
# dataset = 'shopee-iet-machine-learning-competition/' #ok
# dataset = 'shopee-iet/'

makedirs(dataset)
logging_file = os.path.join(dataset ,'summary.log')
filehandler = logging.FileHandler(logging_file)
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)
logging.info(dataset)

target = config_choice(dataset, data_path)
load_dataset = task.Dataset(target['dataset'])

classifier = task.fit(dataset = task.Dataset(target['dataset']),
                      net = target['net'],
                      optimizer = target['optimizer'],
                      epochs = target['epochs'],
                      ngpus_per_trial = target['ngpus_per_trial'],
                      num_trials = target['num_trials'],
                      batch_size = target['batch_size'],
                      verbose=True,
                      plot_results = True)

summary = classifier.fit_summary(output_directory=dataset, verbosity = 3)
logging.info('Top-1 val acc: %.3f' % classifier.results['best_reward'])
logger.info(summary)

# Submit test predictions to Kaggle (Optional)
csv_path = target['dataset'].replace('train', 'sample_submission.csv')
ag.utils.generate_csv_submission(csv_path, dataset, load_dataset, classifier, local_path)

ag.done()
