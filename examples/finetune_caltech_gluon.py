# Run the following to prepare Caltech-256 dataset first:
# $ sh download_caltech.sh
# $ python prepare_caltech.py --data ~/data/caltech/

from __future__ import print_function

import argparse
import random
import os

import mxnet as mx
import numpy as np

import autogluon as ag
from autogluon import autogluon_method

from mxnet import gluon, image, init, nd
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model

# Training settings
parser = argparse.ArgumentParser(description='Caltech-256 Example')
parser.add_argument(
    '--data',
    type=str,
    default='~/data/caltech/',
    help='directory for the prepared data folder')
parser.add_argument(
    '--model',
    type=str,
    default='resnet50_v1b',
    help='name of the pretrained model from model zoo.')
parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    metavar='N',
    help='input batch size for training (default: 64)')
parser.add_argument(
    '--epochs',
    type=int,
    default=1,
    metavar='N',
    help='number of epochs to train (default: 1)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--smoke_test',
    action="store_true",
    help="Finish quickly for testing")
parser.add_argument(
    '--num_gpus',
    default=0,
    type=int,
    help='number of gpus to use, 0 indicates cpu only')
parser.add_argument(
    '--num_workers',
    default=4,
    type=int,
    help='number of preprocessing workers')
parser.add_argument(
    '--classes',
    type=int,
    default=257,
    metavar='N',
    help='number of outputs')
parser.add_argument(
    '--lr',
    default=0.001,
    type=float,
    help='initial learning rate')
parser.add_argument(
    '--momentum',
    default=0.9,
    type=float,
    help='momentum')
parser.add_argument(
    '--wd',
    default=1e-4,
    type=float,
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--lr_factor',
    default=0.75,
    type=float,
    help='learning rate decay ratio')
parser.add_argument(
    '--lr_step',
    default=20,
    type=int,
    help='list of learning rate decay epochs as in str')
parser.add_argument(
    '--expname',
    type=str,
    default='caltechexp')
parser.add_argument(
    '--reuse_actors',
    action="store_true",
    help="reuse actor")
parser.add_argument(
    '--checkpoint_freq',
    default=20, type=int,
    help='checkpoint_freq')
parser.add_argument(
    '--checkpoint_at_end',
    action="store_true",
    help="checkpoint_at_end")
parser.add_argument(
    '--max_failures',
    default=20,
    type=int,
    help='max_failures')
parser.add_argument(
    '--queue_trials',
    action="store_true",
    help="queue_trials")
parser.add_argument(
    '--with_server',
    action="store_true",
    help="with_server")
parser.add_argument(
    '--num_samples',
    type=int,
    default=50,
    metavar='N',
    help='number of samples')
parser.add_argument(
    '--scheduler',
    type=str,
    default='fifo')
args = parser.parse_args()


@autogluon_method
def train_caltech(args):
    # Set Hyper-params
    batch_size = args.batch_size * max(args.num_gpus, 1)
    ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]

    # Define DataLoader
    train_path = os.path.join(args.data, 'train')
    test_path = os.path.join(args.data, 'val')

    jitter_param = 0.4
    lighting_param = 0.1
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    test_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # Load model architecture and Initialize the net with pretrained model
    finetune_net = get_model(args.model, pretrained=True)
    with finetune_net.name_scope():
        finetune_net.fc = nn.Dense(args.classes)
    finetune_net.fc.initialize(init.Xavier(), ctx=ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()

    # Define trainer
    trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
        'learning_rate': args.lr, 'momentum': args.momentum, 'wd': args.wd})
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()

    # Write evaluation loop
    def evaluate_accuracy(data_iterator, net):
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
                outputs = [finetune_net(x) for x in data]
                losses = [L(yhat, y) for yhat, y in zip(outputs, label)]
                autograd.backward(losses)
            loss = 0
            for l in losses:
                loss += l.as_in_context(mx.cpu(0))
            trainer.step(len(data) * data[0].shape[0])
            ##########################
            #  Keep a moving average of the losses
            ##########################
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                           else (
                                        1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

        test_accuracy = evaluate_accuracy(test_data, finetune_net)
        train_accuracy = evaluate_accuracy(train_data, finetune_net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (
            e, moving_loss, train_accuracy, test_accuracy))


if __name__ == "__main__":
    args = parser.parse_args()
    config = {
        'lr': ag.distribution.sample_from(
            lambda: np.power(10.0, np.random.uniform(-4, -1))),
        'momentum': ag.distribution.sample_from(
            lambda: np.random.uniform(0.85, 0.95))
    }
    myscheduler = ag.scheduler.TaskScheduler()
    for i in range(args.num_samples):
        resource = ag.scheduler.Resources(num_cpus=args.num_workers, num_gpus=args.num_gpus)
        task = ag.scheduler.Task(train_caltech, {'args': args, 'config': config}, resource)
        myscheduler.add_task(task)
