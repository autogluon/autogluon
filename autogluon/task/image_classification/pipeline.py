import random
import mxnet as mx
import numpy as np

from mxnet import gluon, init, autograd
from mxnet.gluon import nn
from gluoncv.model_zoo import get_model

__all__ = ['train_image_classification']

def train_image_classification(args, config, reporter):
    vars(args).update(config)
    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)

    # Set Hyper-params
    batch_size = args.batch_size * max(args.num_gpus, 1)
    ctx = [mx.gpu(i)
           for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]

    # Define DataLoader
    train_data = args.train_data
    test_data = args.test_data

    # Load model architecture and Initialize the net with pretrained model
    finetune_net = get_model(args.model, pretrained=args.pretrained)
    with finetune_net.name_scope():
        finetune_net.fc = nn.Dense(args.classes)
    finetune_net.fc.initialize(init.Xavier(), ctx=ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()

    # Define trainer
    trainer = gluon.Trainer(finetune_net.collect_params(), args.optimizer, {
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "wd": args.wd
    })
    L = args.loss
    metric = args.metric

    def train(epoch):
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0],
                                              ctx_list=ctx,
                                              batch_axis=0,
                                              even_split=False)
            label = gluon.utils.split_and_load(batch[1],
                                               ctx_list=ctx,
                                               batch_axis=0,
                                               even_split=False)
            with autograd.record():
                outputs = [finetune_net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(batch_size)
        mx.nd.waitall()

    def test():
        test_loss = 0
        for i, batch in enumerate(test_data):
            data = gluon.utils.split_and_load(batch[0],
                                              ctx_list=ctx,
                                              batch_axis=0,
                                              even_split=False)
            label = gluon.utils.split_and_load(batch[1],
                                               ctx_list=ctx,
                                               batch_axis=0,
                                               even_split=False)
            outputs = [finetune_net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]

            test_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            metric.update(label, outputs)

        _, test_acc = metric.get()
        test_loss /= len(test_data)
        reporter(mean_loss=test_loss, mean_accuracy=test_acc)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()
