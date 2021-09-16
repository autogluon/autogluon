import mxnet as mx

from autogluon.extra.contrib.enas import *

net = ENAS_MBNet()

net.initialize(ctx=mx.gpu(0))
x = mx.nd.random.uniform(shape=(8, 32, 112, 112), ctx=mx.gpu(0))
net.evaluate_latency(x)

print('average latency is ', net.avg_latency)

scheduler = ENAS_Scheduler(net, train_set='imagenet', num_gpus=8, num_cpu=16, controller_type='atten',
                           checkname='./enas_imagenet_nov1/checkpoint.ag')
scheduler.run()

