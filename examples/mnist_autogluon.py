# adapt from http://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html
from __future__ import print_function
import os
import logging
import argparse
import numpy as np

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn

import autogluon as ag
from autogluon import autogluon_method

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

parser = argparse.ArgumentParser(description='AutoGluon MNIST Example')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--epochs', default=5, type=int,
                    help='number of epochs')
parser.add_argument('--num-trials', default=10, type=int,
                    help='number of trail tasks')
parser.add_argument('--scheduler', type=str, default='fifo',
                    help='scheduler name (default: fifo)')
parser.add_argument('--checkpoint', type=str, default='checkpoint/exp1.ag',
                    help='checkpoint path (default: None)')
parser.add_argument('--resume', action='store_true', default= False,
                    help='resume from the checkpoint if needed')
parser.add_argument('--debug', action='store_true', default= False,
                    help='debug if needed')

@autogluon_method
def train_mnist(args, reporter):
    ctx = [mx.gpu(0)]

    # MNIST dataset
    batch_size = 128
    num_outputs = 10
    def transform(data, label):
        return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
    train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                       batch_size, shuffle=True, last_batch='rollover',
                                       num_workers=4)
    test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                      batch_size, shuffle=False, num_workers=4)

    # Define a convolutional neural network
    num_fc = 512
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(in_channels=1, channels=20, kernel_size=5))
        net.add(gluon.nn.BatchNorm(in_channels=20))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(in_channels=20, channels=50, kernel_size=5))
        net.add(gluon.nn.BatchNorm(in_channels=50))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # The Flatten layer collapses all axis, except the first one, into one axis.
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(num_fc,in_units=800))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Dense(num_outputs, in_units=num_fc))

    # Initializae the model wieghts and get Parallel mode
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    # Loss and Optimizer
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': args.lr})

    # Write evaluation loop
    def evaluate_accuracy(data_iterator, net):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            data = gluon.utils.split_and_load(data, ctx_list=ctx)
            label = gluon.utils.split_and_load(label, ctx_list=ctx)
            outputs = [net(x) for x in data]
            predictions = []
            for i, output in enumerate(outputs):
                pred = nd.argmax(output, axis=1)
                acc.update(preds=pred, labels=label[i])
        return acc.get()[1]

    # start training
    smoothing_constant = .01

    for e in range(args.epochs):
        for i, (data, label) in enumerate(train_data):
            data = gluon.utils.split_and_load(data, ctx_list=ctx)
            label = gluon.utils.split_and_load(label, ctx_list=ctx)
            with autograd.record():
                outputs = [net(x) for x in data]
                losses = [softmax_cross_entropy(yhat, y) for yhat, y in zip(outputs, label)]
                autograd.backward(losses)
            loss = 0
            for l in losses:
                loss += l.as_in_context(mx.cpu(0))
            trainer.step(len(data)*data[0].shape[0])
            ##########################
            #  Keep a moving average of the losses
            ##########################
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

        test_accuracy = evaluate_accuracy(test_data, net)
        reporter(epoch=e, accuracy=test_accuracy)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # creating hyperparameters
    cs = CS.ConfigurationSpace()
    lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1e-1, log=True)
    cs.add_hyperparameter(lr)

    # create searcher and scheduler
    searcher = ag.searcher.RandomSampling(cs)
    if args.scheduler == 'hyperband':
        myscheduler = ag.scheduler.Hyperband_Scheduler(train_mnist, args,
                                                       {'num_cpus': 2, 'num_gpus': 1}, searcher,
                                                       num_trials=args.num_trials,
                                                       checkpoint=args.checkpoint,
                                                       resume = args.resume,
                                                       time_attr='epoch', reward_attr="accuracy",
                                                       max_t=args.epochs, grace_period=1)
    else:
        myscheduler = ag.scheduler.FIFO_Scheduler(train_mnist, args,
                                                  {'num_cpus': 2, 'num_gpus': 1}, searcher,
                                                  num_trials=args.num_trials,
                                                  checkpoint=args.checkpoint,
                                                  resume = args.resume,
                                                  reward_attr="accuracy")

    myscheduler.run()
    myscheduler.join_tasks()
    myscheduler.get_training_curves('{}.png'.format(os.path.splitext(args.checkpoint)[0]))

    print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                                   myscheduler.get_best_reward()))
