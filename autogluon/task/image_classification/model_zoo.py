import gluoncv as cv

from autogluon.network import autogluon_nets, autogluon_net_instances, Net

__all__ = ['models', 'get_model', 'get_model_instances']

models = ['resnet18_v1', 'resnet34_v1',
          'resnet50_v1', 'resnet101_v1', 'resnet152_v1',
          'resnet18_v1b', 'resnet34_v1b', 'resnet50_v1b_gn',
          'resnet50_v1b', 'resnet101_v1b', 'resnet152_v1b',
          'resnet50_v1c', 'resnet101_v1c', 'resnet152_v1c',
          'resnet50_v1d', 'resnet101_v1d', 'resnet152_v1d',
          'resnet50_v1e', 'resnet101_v1e', 'resnet152_v1e',
          'resnet18_v2', 'resnet34_v2',
          'resnet50_v2', 'resnet101_v2', 'resnet152_v2',
          'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d',
          'se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnext101_64x4d',
          'se_resnet18_v1', 'se_resnet34_v1', 'se_resnet50_v1',
          'se_resnet101_v1', 'se_resnet152_v1',
          'se_resnet18_v2', 'se_resnet34_v2', 'se_resnet50_v2',
          'se_resnet101_v2', 'se_resnet152_v2',
          'senet_154', 'squeezenet1.0', 'squeezenet1.1',
          'mobilenet1.0', 'mobilenet0.75', 'mobilenet0.5', 'mobilenet0.25',
          'mobilenetv2_1.0', 'mobilenetv2_0.75', 'mobilenetv2_0.5', 'mobilenetv2_0.25',
          'densenet121', 'densenet161', 'densenet169', 'densenet201',
          'darknet53', 'alexnet',
          'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
          'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
          'residualattentionnet56', 'residualattentionnet92',
          'residualattentionnet128', 'residualattentionnet164',
          'residualattentionnet200', 'residualattentionnet236', 'residualattentionnet452',
          'resnet18_v1b_0.89', 'resnet50_v1d_0.86', 'resnet50_v1d_0.48',
          'resnet50_v1d_0.37', 'resnet50_v1d_0.11',
          'resnet101_v1d_0.76', 'resnet101_v1d_0.73']


@autogluon_net_instances
def get_model_instances(name, **kwargs):
    name = name.lower()
    if name not in models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(models)))
        raise ValueError(err_str)
    net = cv.model_zoo._models[name](**kwargs)
    return net


@autogluon_nets
def get_model(name, **kwargs):
    """Returns a network with search space by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    Net
        The model with search space.
    """
    name = name.lower()
    if name not in models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(models)))
        raise ValueError(err_str)
    net = Net(name)
    return net

#TODO (cgraywang): add more models using method


@autogluon_nets
def resnet18_v1(**kwargs):
    pass


@autogluon_nets
def resnet34_v1(**kwargs):
    pass


@autogluon_nets
def resnet50_v1(**kwargs):
    pass


@autogluon_nets
def resnet101_v1(**kwargs):
    pass


@autogluon_nets
def resnet152_v1(**kwargs):
    pass


@autogluon_nets
def resnet18_v1b(**kwargs):
    pass


@autogluon_nets
def resnet34_v1b(**kwargs):
    pass


@autogluon_nets
def resnet50_v1b_gn(**kwargs):
    pass


@autogluon_nets
def resnet50_v1b(**kwargs):
    pass


@autogluon_nets
def resnet101_v1b(**kwargs):
    pass


@autogluon_nets
def resnet152_v1b(**kwargs):
    pass


@autogluon_nets
def resnet50_v1c(**kwargs):
    pass


@autogluon_nets
def resnet101_v1c(**kwargs):
    pass


@autogluon_nets
def resnet152_v1c(**kwargs):
    pass


@autogluon_nets
def resnet50_v1d(**kwargs):
    pass


@autogluon_nets
def resnet101_v1d(**kwargs):
    pass


@autogluon_nets
def resnet152_v1d(**kwargs):
    pass


@autogluon_nets
def resnet50_v1e(**kwargs):
    pass


@autogluon_nets
def resnet101_v1e(**kwargs):
    pass


@autogluon_nets
def resnet152_v1e(**kwargs):
    pass


@autogluon_nets
def resnet18_v2(**kwargs):
    pass


@autogluon_nets
def resnet34_v2(**kwargs):
    pass


@autogluon_nets
def resnet50_v2(**kwargs):
    pass


@autogluon_nets
def resnet101_v2(**kwargs):
    pass


@autogluon_nets
def resnet152_v2(**kwargs):
    pass


@autogluon_nets
def resnext50_32x4d(**kwargs):
    pass


@autogluon_nets
def resnext101_32x4d(**kwargs):
    pass


@autogluon_nets
def resnext101_64x4d(**kwargs):
    pass


@autogluon_nets
def se_resnext50_32x4d(**kwargs):
    pass


@autogluon_nets
def se_resnext101_32x4d(**kwargs):
    pass


@autogluon_nets
def se_resnext101_64x4d(**kwargs):
    pass


@autogluon_nets
def se_resnet18_v1(**kwargs):
    pass


@autogluon_nets
def se_resnet34_v1(**kwargs):
    pass


@autogluon_nets
def se_resnet50_v1(**kwargs):
    pass


@autogluon_nets
def se_resnet101_v1(**kwargs):
    pass


@autogluon_nets
def se_resnet152_v1(**kwargs):
    pass


@autogluon_nets
def se_resnet18_v2(**kwargs):
    pass


@autogluon_nets
def se_resnet34_v2(**kwargs):
    pass


@autogluon_nets
def se_resnet50_v2(**kwargs):
    pass


@autogluon_nets
def se_resnet101_v2(**kwargs):
    pass


@autogluon_nets
def se_resnet152_v2(**kwargs):
    pass


@autogluon_nets
def senet_154(**kwargs):
    pass


@autogluon_nets
def densenet121(**kwargs):
    pass


@autogluon_nets
def densenet161(**kwargs):
    pass


@autogluon_nets
def densenet169(**kwargs):
    pass


@autogluon_nets
def densenet201(**kwargs):
    pass


@autogluon_nets
def darknet53(**kwargs):
    pass


@autogluon_nets
def alexnet(**kwargs):
    pass


@autogluon_nets
def vgg11(**kwargs):
    pass


@autogluon_nets
def vgg11_bn(**kwargs):
    pass


@autogluon_nets
def vgg13(**kwargs):
    pass


@autogluon_nets
def vgg13_bn(**kwargs):
    pass


@autogluon_nets
def vgg16(**kwargs):
    pass


@autogluon_nets
def vgg16_bn(**kwargs):
    pass


@autogluon_nets
def vgg19(**kwargs):
    pass


@autogluon_nets
def vgg19_bn(**kwargs):
    pass


@autogluon_nets
def residualattentionnet56(**kwargs):
    pass


@autogluon_nets
def residualattentionnet92(**kwargs):
    pass


@autogluon_nets
def residualattentionnet128(**kwargs):
    pass


@autogluon_nets
def residualattentionnet164(**kwargs):
    pass


@autogluon_nets
def residualattentionnet200(**kwargs):
    pass


@autogluon_nets
def residualattentionnet236(**kwargs):
    pass


@autogluon_nets
def residualattentionnet452(**kwargs):
    pass
