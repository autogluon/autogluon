import os
import argparse
import logging
import autogluon as ag
from autogluon import ImageClassification as task
from kaggle_configuration import config_choice
"""
--data-dir
data
--dataset
shopee-iet
--num-trials
2
--num-epochs
1
--batch-size
8
--ngpus-per-trial
1

python benchmark.py --data-dir data --dataset shopee-iet --num-trials 2 --num-epochs 1 --batch-size 8 --ngpus-per-trial 1


# ensemble=1
# test
python benchmark.py --data-dir data --dataset dog-breed-identification --num-trials 3 --num-epochs 1 --batch-size 48 --ngpus-per-trial 1 

# 10_5 # succeed
python benchmark.py --data-dir data --dataset dog-breed-identification --num-trials 5 --num-epochs 10 --batch-size 48 --ngpus-per-trial 1

# test ensemble / submission
python benchmark.py --data-dir data --dataset dog-breed-identification --num-trials 1 --num-epochs 1 --batch-size 48 --ngpus-per-trial 1 --submission

# ensemble=2 的两次相同？ 必须要num-trials >1 ? 
# test_ensemble=2 # failure
python benchmark.py --data-dir data --dataset dog-breed-identification --num-trials 1 --num-epochs 2 --batch-size 48 --ngpus-per-trial 1 --resume --submission


# 
python benchmark.py --data-dir data --dataset dog-breed-identification --num-trials 1 --num-epochs 180 --batch-size 48 --ngpus-per-trial 1 --resume --submission

"""
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for different kaggle competitions.')
    parser.add_argument('--data-dir', type=str, default='data/',
                        help='training and validation pictures to use.')
    parser.add_argument('--dataset', type=str, default='shopee-iet',
                        help='the kaggle competition.')
    parser.add_argument('--custom', type=str, default = 'predict',
                        help='the name of the submission file you set.')
    parser.add_argument('--num-trials', type=int, default=30,
                        help='number of trials')
    parser.add_argument('--num-epochs', type=int, default=5,
                        help='number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--ngpus-per-trial', type=int, default=1,
                        help='number of gpus to use.')
    parser.add_argument('--resume', action='store_true',
                        help='whether to load last hyperparameters to retrain.')
    parser.add_argument('--submission', action='store_true',
                        help='whether to submit test predictions to Leaderboard (Optional).')
    opt = parser.parse_args()
    return opt

def predict_details(data_path, classifier, load_dataset):
    test_dataset = os.path.join(data_path ,'test')
    inds, probs, probs_all= classifier.predict(task.Dataset(test_dataset))
    value = []
    target_dataset = load_dataset.init()
    for i in inds:
        value.append(target_dataset.classes[i])
    return inds, probs, probs_all, value

def main():
    opt = parse_args()
    if not os.path.exists(opt.dataset):
        os.mkdir(opt.dataset)
    local_path = os.path.dirname(__file__)
    data_path = os.path.join(local_path, opt.data_dir, opt.dataset)
    output_directory = os.path.join(opt.dataset ,'checkpoint/')
    filehandler = logging.FileHandler(os.path.join(opt.dataset ,'summary.log'))
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logging.info(opt)

    target = config_choice(opt.dataset, opt.data_dir)
    load_dataset = task.Dataset(target['dataset'])
    classifier = task.fit(dataset = load_dataset,
                          output_directory = output_directory,
                          net = target['net'],
                          optimizer = target['optimizer'],
                          tricks = target['tricks'],
                          lr_config = target['lr_config'],
                          resume=opt.resume,
                          epochs=opt.num_epochs,
                          ngpus_per_trial=opt.ngpus_per_trial,
                          num_trials=opt.num_trials,
                          batch_size=opt.batch_size,
                          ensemble=2,
                          verbose=True,
                          plot_results = True)

    summary = classifier.fit_summary(output_directory=opt.dataset, verbosity=4)
    logging.info('Top-1 val acc: %.3f' % classifier.results['best_reward'])
    logger.info(summary)

    if opt.submission:
        inds, probs, probs_all, value = predict_details(data_path, classifier, load_dataset)
        # logging.info(inds)
        # logging.info(probs)
        # logging.info(probs_all)
        # logging.info(value)
        ag.utils.generate_csv_submission(opt.dataset, data_path, local_path, inds, probs_all, value, opt.custom)

if __name__ == '__main__':
    main()
