import os
import time
import logging
import numpy as np

import mxnet as mx
from mxnet import gluon, init
from gluoncv.model_zoo import get_model

import autogluon as ag
from autogluon import autogluon_register_args
from autogluon.utils.mxutils import get_data_rec

@autogluon_register_args(
    dataset='apparel',
    resume=False,
    epochs=ag.ListSpace(80, 40, 120),
    lr=ag.LogLinearSpace(1e-4, 1e-2),
    lr_factor=ag.LogLinearSpace(0.1, 1),
    batch_size=256,
    momentum=0.9,
    wd=ag.LogLinearSpace(1e-5, 1e-3),
    num_gpus=8,
    num_workers=30,
    input_size=ag.ListSpace(224, 256),
    crop_ratio=0.875,
    jitter_param=ag.LinearSpace(0.1, 0.4),
    max_rotate_angle=ag.IntSpace(0, 10),
    remote_file='remote_ips.txt',
)
def train_finetune(args, reporter):
    lr_steps = [int(args.epochs*0.75), np.inf]
    ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]

    num_classes, rec_train, rec_train_idx, rec_val, rec_val_idx = get_dataset_meta(args.dataset)
    finetune_net = get_network(num_classes, ctx)

    train_data, val_data, batch_fn = get_data_rec(
            args.input_size, args.crop_ratio, rec_train, rec_train_idx,
            rec_val, rec_val_idx, args.batch_size, args.num_workers,
            args.jitter_param, args.max_rotate_angle)

    trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
                            'learning_rate': args.lr, 'momentum': args.momentum, 'wd': args.wd})
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()

    lr_counter = 0
    for epoch in range(args.epochs):
        if epoch == lr_steps[lr_counter]:
            print('Decreasing LR to ', trainer.learning_rate*args.lr_factor)
            trainer.set_learning_rate(trainer.learning_rate*args.lr_factor)
            lr_counter += 1

        train_data.reset()
        metric.reset()
        for i, batch in enumerate(train_data):
            data, label = batch_fn(batch, ctx)
            with mx.autograd.record():
                outputs = [finetune_net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(args.batch_size)
            metric.update(label, outputs)

        _, train_acc = metric.get()
        _, val_acc = test(finetune_net, val_data, ctx, batch_fn)

        if reporter is not None:
            reporter(epoch=epoch, accuracy=val_acc)
        else:
            print('[Epoch %d] Train-acc: %.3f | Val-acc: %.3f' %
                  (epoch, train_acc, val_acc))

def test(net, val_data, ctx, batch_fn):
    metric = mx.metric.Accuracy()
    val_data.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()

def get_network(num_classes, ctx):
    finetune_net = get_model('densenet169', pretrained=False)
    finetune_net.collect_params().load('densenet169-0000.params')
    with finetune_net.name_scope():
        finetune_net.output = gluon.nn.Dense(num_classes)
    finetune_net.output.initialize(init.Xavier(), ctx=ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()
    return finetune_net

def get_dataset_meta(dataset, basedir='./datasets', final_fit=False):
    if final_fit:
        if dataset.lower() == 'apparel':
            num_classes = 18
            rec_train = os.path.join(basedir, 'Apparel_train.rec')
            rec_train_idx = os.path.join(basedir, 'Apparel_train.idx')
            rec_val = os.path.join(basedir, 'Apparel_test.rec')
            rec_val_idx = os.path.join(basedir, 'Apparel_test.idx')
        elif dataset.lower() == 'footwear':
            num_classes = 19
            rec_train = os.path.join(basedir, 'Footwear_train.rec')
            rec_train_idx = os.path.join(basedir, 'Footwear_train.idx')
            rec_val = os.path.join(basedir, 'Footwear_test.rec')
            rec_val_idx = os.path.join(basedir, 'Footwear_test.idx')
        elif dataset.lower() == 'landmarks':
            num_classes = 20
            rec_train = os.path.join(basedir, 'Landmarks_train.rec')
            rec_train_idx = os.path.join(basedir, 'Landmarks_train.idx')
            rec_val = os.path.join(basedir, 'Landmarks_test.rec')
            rec_val_idx = os.path.join(basedir, 'Landmarks_test.idx')
        elif dataset.lower() == 'weapons':
            num_classes = 11
            rec_train = os.path.join(basedir, 'Weapons_train.rec')
            rec_train_idx = os.path.join(basedir, 'Weapons_train.idx')
            rec_val = os.path.join(basedir, 'Weapons_test.rec')
            rec_val_idx = os.path.join(basedir, 'Weapons_test.idx')
        else:
            raise NotImplemented
    else:
        if dataset.lower() == 'apparel':
            num_classes = 18
            rec_train = os.path.join(basedir, 'Apparel_train_split.rec')
            rec_train_idx = os.path.join(basedir, 'Apparel_train_split.idx')
            rec_val = os.path.join(basedir, 'Apparel_val_split.rec')
            rec_val_idx = os.path.join(basedir, 'Apparel_val_split.idx')
        elif dataset.lower() == 'footwear':
            num_classes = 19
            rec_train = os.path.join(basedir, 'Footwear_train_split.rec')
            rec_train_idx = os.path.join(basedir, 'Footwear_train_split.idx')
            rec_val = os.path.join(basedir, 'Footwear_val_split.rec')
            rec_val_idx = os.path.join(basedir, 'Footwear_val_split.idx')
        elif dataset.lower() == 'landmarks':
            num_classes = 20
            rec_train = os.path.join(basedir, 'Landmarks_train_split.rec')
            rec_train_idx = os.path.join(basedir, 'Landmarks_train_split.idx')
            rec_val = os.path.join(basedir, 'Landmarks_val_split.rec')
            rec_val_idx = os.path.join(basedir, 'Landmarks_val_split.idx')
        elif dataset.lower() == 'weapons':
            num_classes = 11
            rec_train = os.path.join(basedir, 'Weapons_train_split.rec')
            rec_train_idx = os.path.join(basedir, 'Weapons_train_split.idx')
            rec_val = os.path.join(basedir, 'Weapons_val_split.rec')
            rec_val_idx = os.path.join(basedir, 'Weapons_val_split.idx')
        else:
            raise NotImplemented
        
    return num_classes, rec_train, rec_train_idx, rec_val, rec_val_idx

if __name__ == '__main__':
    start_time = time.time()
    args = train_finetune.args
    args.dataset = 'footwear'
    """
    # if you want to launch single training, do it like this:
    best_config = {'epochs': 40, 'input_size': 256, 'jitter_param': 0.293152307944186,
                   'lr': 0.005240915838601111, 'lr_factor': 0.9343588751671241,
                   'max_rotate_angle': 4, 'wd': 2.886850482040357e-05, 'final_fit':True}
    train_finetune(args, best_config, reporter=None)
    """
    logging.basicConfig(level=logging.DEBUG)
    # create searcher and scheduler
    searcher = ag.searcher.RandomSampling(train_finetune.cs)
    myscheduler = ag.distributed.DistributedHyperbandScheduler(train_finetune, args,
                                                               resource={'num_cpus': 16, 'num_gpus': args.num_gpus},
                                                               searcher=searcher,
                                                               checkpoint='./{}/checkerpoint.ag'.format(args.dataset),
                                                               num_trials=64,
                                                               resume=args.resume,
                                                               time_attr='epoch',
                                                               reward_attr="accuracy",
                                                               max_t=120,
                                                               grace_period=4)
    myscheduler.run()
    myscheduler.join_tasks()
    myscheduler.get_training_curves('{}.png'.format(args.dataset))

    print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                                   myscheduler.get_best_reward()))
    print('Costed time: {}'.format(time.time() - start_time))

    best_config = myscheduler.get_best_config()
    best_config['final_fit'] = True
    train_finetune(args, best_config, reporter=None)

    myscheduler.shutdown()
