import mxnet as mx

from .nets import get_network


class Sample_params(object):
    propose = "Sample params"

    def __init__(self, *args):
        batch_size, num_gpus, num_workers = args
        self._batch_size = batch_size * max(1, num_gpus)
        self._num_gpus = num_gpus
        self._context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

    @classmethod
    def tell_info(cls):
        print("propose:", cls.propose)

    @property
    def get_batchsize(self):
        return self._batch_size

    @property
    def get_context(self):
        return self._context


class Getmodel_kwargs:
    def __init__(self,
                 context,
                 classes,
                 model_name,
                 model_teacher,
                 hard_weight,
                 hybridize,
                 multi_precision=False,
                 use_pretrained=True,
                 use_gn=False,
                 last_gamma=False,
                 batch_norm=False,
                 use_se=False):
        self._kwargs = {'ctx': context, 'pretrained': use_pretrained, 'num_classes': classes}
        self._model_name = model_name
        self._model_teacher = model_teacher
        self._hybridize = hybridize
        self._hard_weight = hard_weight

        if multi_precision:
            self._dtype = 'float16'
        else:
            self._dtype = 'float32'

        if use_gn:
            from gluoncv.nn import GroupNorm
            self._kwargs['norm_layer'] = GroupNorm

        if isinstance(model_name, str):
            if model_name.startswith('vgg'):
                self._kwargs['batch_norm'] = batch_norm
            elif model_name.startswith('resnext'):
                self._kwargs['use_se'] = use_se

        if last_gamma:
            self._kwargs['last_gamma'] = True

        if self._model_teacher is not None and self._hard_weight < 1.0:
            self.distillation = True
        else:
            self.distillation = False

    @property
    def dtype(self):
        return self._dtype

    @property
    def get_teacher(self):
        net = get_network(self._model_teacher, **self._kwargs)
        net.cast(self._dtype)
        if self._hybridize:
            net.hybridize(static_alloc=True, static_shape=True)
        return net

    @property
    def get_net(self):
        net = get_network(self._model_name, **self._kwargs)
        net.cast(self._dtype)
        if self._hybridize:
            net.hybridize(static_alloc=True, static_shape=True)
        return net
