import os, glob
import argparse
import autogluon as ag
import logging
from autogluon import ImageClassification as task
from autogluon import config_choice
from gluoncv.utils import makedirs

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for different kaggle competitions.')
    parser.add_argument('--data-dir', type=str, default='/home/ubuntu/workspace/data/dataset/',
                        help='training and validation pictures to use.')
    parser.add_argument('--dataset', type=str, default='dogs-vs-cats-redux-kernels-edition',
                        help='the kaggle competition')
    opt = parser.parse_args()
    return opt
opt = parse_args()

# data
local_path = os.path.dirname(__file__)
makedirs(opt.dataset)
logging_file = os.path.join(opt.dataset ,'summary.log')
filehandler = logging.FileHandler(logging_file)
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)
logging.info(opt.dataset)

target = config_choice(opt.dataset, opt.data_dir)
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

summary = classifier.fit_summary(output_directory=opt.dataset, verbosity = 3)
logging.info('Top-1 val acc: %.3f' % classifier.results['best_reward'])
logger.info(summary)

# Submit test predictions to Kaggle (Optional)
csv_path = target['dataset'].replace('train', 'sample_submission.csv')
ag.utils.generate_csv_submission(csv_path, opt.dataset, load_dataset, classifier, local_path)

ag.done()
