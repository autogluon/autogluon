import logging
import argparse
import os
import random
import numpy as np
import mxnet as mx

import autogluon as ag
from autogluon import TextClassification as task

parser = argparse.ArgumentParser(description='AutoGluon text classification')
parser.add_argument('--data', default='SST', type=str,
                    help='options are SST, MRPC, QQP, QNLI, RTE, STS-B, CoLA, MNLI, WNLI, IMDB')
parser.add_argument('--nets', default='bert_12_768_12', type=str,
                    help='list of nets to run (default: bert_12_768_12)')
parser.add_argument('--optims', default='bertadam', type=str,
                    help='list of optims to run, (default: bertadam)')
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
parser.add_argument('--epochs', default=10, type=int,
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

    logger = logging.getLogger(__name__)
    logdir = os.path.join(os.path.splitext(args.savedir)[0].split('/')[0], 'log')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    dataset = task.Dataset(name=args.data)
    results = task.fit(dataset,
                       net=ag.Choice(args.nets),
                       time_limits=args.time_limits,
                       epochs=args.epochs,
                       num_trials=args.num_trials,
                       accumulate=args.accumulate,
                       batch_size=args.batch_size,
                       bert_dataset=args.bert_dataset,
                       dev_batch_size=args.dev_batch_size, gpu=args.gpu,
                       log_interval=args.log_interval,
                       lr=ag.LogLinear(2e-06, 2e-04), #2e-05
                       max_len=args.max_len,
                       model_parameters=args.model_parameters, optimizer=ag.Choice(args.optims),
                       output_dir=args.savedir, pretrained_bert_parameters=args.pretrained_bert_parameters,
                       seed=args.seed, warmup_ratio=args.warmup_ratio, epsilon=args.epsilon,
                       dtype=args.dtype, only_inference=args.only_inference, pad=args.pad,
                       visualizer=args.visualizer)

    print('Top-1 val acc: %.3f' % results.reward)

    # test_acc = task.evaluate(dataset)
    # print('Top-1 test acc: %.3f' % test_acc)
    #
    # sentence = 'I feel this is awesome!'
    # ind, prob = task.predict(sentence)
    # print('The input sentence is classified as [%s], with probability %.2f.' %
    #       (dataset.train.synsets[ind.asscalar()], prob.asscalar()))
    #
    # print('The best configuration is:')
    # print(results.config)