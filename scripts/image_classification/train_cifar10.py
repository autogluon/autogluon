import logging
import argparse
import os
import random
import numpy as np
import mxnet as mx

from gluoncv.data import transforms as gcv_transforms
from mxnet.gluon.data.vision import transforms

import autogluon as ag
from autogluon import image_classification as task

parser = argparse.ArgumentParser(description='AutoGluon CIFAR10')
parser.add_argument('--data', default='CIFAR10', type=str,
                    help='options are: cifar10 or mnist')
parser.add_argument('--nets', default='resnet18_v1,resnet34_v1', type=str,
                    help='list of nets to run (default: resnet18_v1,resnet34_v1)')
parser.add_argument('--optims', default='sgd,adam,nag', type=str,
                    help='list of optims to run, (default: sgd,adam,nag)')
parser.add_argument('--searcher', type=str, default='random',
                    help='searcher name (default: random)')
parser.add_argument('--trial_scheduler', type=str, default='fifo',
                    help='trial scheduler name (options: fifo or hyperband)')
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch_size')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume from the checkpoint if needed')
parser.add_argument('--savedir', type=str, default='checkpoint/exp1.ag',
                    help='save and checkpoint path (default: None)')
parser.add_argument('--visualizer', type=str, default='tensorboard',
                    help='visualizer (default: tensorboard)')
parser.add_argument('--time_limits', default=1 * 60 * 60, type=int,
                    help='time limits in seconds')
parser.add_argument('--max_metric', default=1.0, type=float,
                    help='the max metric that is used to stop the trials')
parser.add_argument('--num_trials', default=6, type=int,
                    help='number of experiment trials')
parser.add_argument('--num_gpus', default=1, type=int,
                    help='number of gpus per trial')
parser.add_argument('--max_num_cpus', default=4, type=int,
                    help='number of cpus per trial')
parser.add_argument('--num_training_epochs', default=10, type=int,
                    help='number of epochs per trial')
parser.add_argument('--lr_factor', default=0.75, type=float,
                    help='learning rate decay ratio')
parser.add_argument('--lr_step', default=20, type=int,
                    help='list of learning rate decay epochs as in str')
parser.add_argument('--demo', action='store_true', default=False,
                    help='demo if needed')
parser.add_argument('--debug', action='store_true', default=False,
                    help='debug if needed')
parser.add_argument('--seed', default=100, type=int,
                    help='random seed')

if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)

    logger = logging.getLogger(__name__)
    logdir = os.path.join(os.path.splitext(args.savedir)[0].split('/')[0], 'log')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if args.debug:
        logging.basicConfig(filename='%s/train.log' % logdir,
                            filemode='a', level=logging.DEBUG)
    else:
        logging.basicConfig(filename='%s/train.log' % logdir,
                            filemode='a', level=logging.INFO)

    transform_train_list = [
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ]
    transform_val_list = [
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ]
    dataset = task.Dataset(name=args.data, batch_size=args.batch_size,
                           transform_train_list=transform_train_list,
                           transform_val_list=transform_val_list)
    net_list = [net for net in args.nets.split(',')]
    optim_list = [opt for opt in args.optims.split(',')]
    stop_criterion = {
        'time_limits': args.time_limits,
        'max_metric': args.max_metric,
        'num_trials': args.num_trials
    }
    resources_per_trial = {
        'num_gpus': args.num_gpus,
        'num_training_epochs': args.num_training_epochs

    }

    results = task.fit(dataset,
                       nets=ag.Nets(net_list),
                       optimizers=ag.Optimizers(optim_list),
                       searcher=args.searcher,
                       trial_scheduler=args.trial_scheduler,
                       resume=args.resume,
                       savedir=args.savedir,
                       visualizer=args.visualizer,
                       stop_criterion=stop_criterion,
                       resources_per_trial=resources_per_trial,
                       lr_factor=args.lr_factor,
                       lr_step=args.lr_step,
                       demo=args.demo)

    logger.info('Top-1 acc: %.2f' % (results.metric * 100))
    test_acc = task.evaluate()
    logger.info('Top-1 test acc: %.2f' % (test_acc * 100))
    logger.info('Time: %.2f s' % results.time)
