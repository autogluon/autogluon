"""Pretrained models for Standford Dogs dataset"""
import os
import mxnet as mx
import gluoncv as gcv
from ..model_store import get_model_file

__all__ = ['standford_dog_resnet152_v1', 'standford_dog_resnext101_64x4d']

def standford_dog_resnet152_v1(pretrained=False, root=os.path.join('~', '.autogluon', 'models'),
                               ctx=mx.cpu(0), **kwargs):
    net = gcv.model_zoo.resnet152_v1(classes=120, **kwargs)
    if pretrained:
        net.load_parameters(get_model_file('standford_dog_resnet152_v1',
                                           root=root), ctx=ctx)
    return net

def standford_dog_resnext101_64x4d(pretrained=False, root=os.path.join('~', '.autogluon', 'models'),
                                   ctx=mx.cpu(0), **kwargs):
    net = gcv.model_zoo.resnext.resnext101_64x4d(classes=120, **kwargs)
    if pretrained:
        net.load_parameters(get_model_file('standford_dog_resnext101_64x4d',
                                           root=root), ctx=ctx)
    return net
