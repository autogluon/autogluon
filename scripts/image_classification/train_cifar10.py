import logging
import argparse
import os

import autogluon as ag
from autogluon import image_classification as task

parser = argparse.ArgumentParser(description='AutoGluon CIFAR10')
parser.add_argument('--data', default='CIFAR10', type=str,
                    help='options are: cifar10 or mnist')
parser.add_argument('--nets', default='cifar_resnet20_v1', type=str,
                    help='list of nets to run (default: cifar_resnet20_v1)')
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
parser.add_argument('--max_trial_count', default=6, type=int,
                    help='number of experiment trials')
parser.add_argument('--max_num_gpus', default=1, type=int,
                    help='number of gpus per trial')
parser.add_argument('--max_num_cpus', default=4, type=int,
                    help='number of cpus per trial')
parser.add_argument('--max_training_epochs', default=10, type=int,
                    help='number of epochs per trial')
parser.add_argument('--lr_factor', default=0.75, type=float,
                    help='learning rate decay ratio')
parser.add_argument('--lr_step', default=20, type=int,
                    help='list of learning rate decay epochs as in str')
parser.add_argument('--demo', action='store_true', default=False,
                    help='demo if needed')
parser.add_argument('--debug', action='store_true', default=False,
                    help='debug if needed')

if __name__ == '__main__':
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logging_handlers = [logging.StreamHandler()]
    logdir = os.path.join(os.path.splitext(args.savedir)[0], 'logs')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging_handlers.append(logging.FileHandler(
        '%s/train_cifar10.log' % logdir))

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, handlers=logging_handlers)
    else:
        logging.basicConfig(level=logging.INFO, handlers=logging_handlers)

    dataset = task.Dataset(name=args.data, batch_size=args.batch_size)
    net_list = [net for net in args.nets.split(',')]
    optim_list = [opt for opt in args.optims.split(',')]
    stop_criterion = {
        'time_limits': args.time_limits,
        'max_metric': args.max_metric,
        'max_trial_count': args.max_trial_count
    }
    resources_per_trial = {
        'max_num_gpus': args.max_num_gpus,
        'max_num_cpus': args.max_num_cpus,
        'max_training_epochs': args.max_training_epochs
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

    logger.info('Best result:')
    logger.info(results.metric)
    logger.info('=========================')
    logger.info('Best search space:')
    logger.info(results.config)
    logger.info('=========================')
    logger.info('Total time cost:')
    logger.info(results.time)
    logger.info('=========================')
    logger.info('Finished!')
