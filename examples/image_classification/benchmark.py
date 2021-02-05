import os
import argparse
import logging
import pandas as pd
from autogluon.core.utils import generate_csv_submission
from autogluon.vision import ImagePredictor
from kaggle_configuration import config_choice

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for different kaggle competitions.')
    parser.add_argument('--data-dir', type=str, default='data/',
                        help='training and validation pictures to use.')
    parser.add_argument('--dataset', type=str, default='shopee-iet',
                        help='the kaggle competition.')
    parser.add_argument('--custom', type=str, default='predict',
                        help='the name of the submission file you set.')
    parser.add_argument('--num-trials', type=int, default=-1,
                        help='number of trials, if negative, use default setting')
    parser.add_argument('--num-epochs', type=int, default=-1,
                        help='number of training epochs, if negative, will use default setting')
    parser.add_argument('--batch-size', type=int, default=-1,
                        help='training batch size per device (CPU/GPU). If negative, will use default setting')
    parser.add_argument('--ngpus-per-trial', type=int, default=1,
                        help='number of gpus to use.')
    opt = parser.parse_args()
    return opt

def predict_details(test_dataset, predictor):
    res = predictor.predict(test_dataset)
    inds, probs, value = res['id'], res['score'], res['class']
    res_prob = predictor.predict_proba(test_dataset)
    probs_all = res_prob['score']
    return inds.tolist(), probs.tolist(), probs_all.tolist(), value.tolist()

def main():
    opt = parse_args()
    if not os.path.exists(opt.dataset):
        os.mkdir(opt.dataset)

    local_path = '.'
    output_directory = os.path.join(opt.dataset, 'checkpoint/')
    filehandler = logging.FileHandler(os.path.join(opt.dataset, 'summary.log'))
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logging.info(opt)

    target_hyperparams = config_choice(opt.data_dir, opt.dataset)
    dataset_dir = target_hyperparams.pop('dataset')
    train_dataset, val_dataset, test_dataset = ImagePredictor.Dataset.from_folders(dataset_dir)
    if isinstance(val_dataset, pd.DataFrame) and len(val_dataset) < 1:
        val_dataset = None
    predictor = ImagePredictor(log_dir=output_directory)
    num_classes = target_hyperparams.pop('classes')
    assert num_classes == len(train_dataset.classes), f'num_class mismatch, {num_classes} vs {len(train_dataset.classes)}'
    # overwriting default by command line:
    if opt.batch_size > 0:
        target_hyperparams['batch_size'] = opt.batch_size
    num_trials = target_hyperparams.pop('num_trials')
    ngpus_per_trial = target_hyperparams.pop('ngpus_per_trial')
    if opt.num_epochs > 0:
        target_hyperparams['epochs'] = opt.num_epochs
    num_trials = opt.num_trials if opt.num_trials > 0 else num_trials
    if opt.ngpus_per_trial >= 0:
        ngpus_per_trial = opt.ngpus_per_trial
    predictor.fit(train_data=train_dataset,
                  val_data=val_dataset,
                  hyperparameters=target_hyperparams,
                  ngpus_per_trial=ngpus_per_trial,
                  num_trials=num_trials,
                  verbosity=2)

    summary = predictor.fit_summary()
    logging.info('Top-1 val acc: %.3f' % summary['train_acc'])
    logger.info(summary)

if __name__ == '__main__':
    main()
