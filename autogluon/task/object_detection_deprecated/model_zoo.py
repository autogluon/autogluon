from mxnet import gluon
import gluoncv

from ...network import autogluon_nets, autogluon_net_instances, Net

__all__ = ['models', 'get_model', 'get_model_instances', 'get_norm_layer']

models = ['ssd_300_vgg16_atrous_coco', 'ssd_512_vgg16_atrous_coco', 'ssd_512_resnet50_v1_coco',
          'ssd_512_mobilenet1.0_coco',
          'faster_rcnn_resnet50_v1b_coco', 'faster_rcnn_resnet101_v1d_coco',
          'faster_rcnn_fpn_resnet50_v1b_coco',
          'faster_rcnn_fpn_resnet101_v1d_coco', 'faster_rcnn_fpn_bn_resnet50_v1b_coco',
          'faster_rcnn_fpn_bn_resnet50_v1b_coco', 'yolo3_darknet53_coco', 'yolo3_darknet53_coco',
          'yolo3_mobilenet1.0_coco']

norm_layers = {'BatchNorm': gluon.nn.BatchNorm,
               'SyncBatchNorm': gluon.contrib.nn.SyncBatchNorm}


def get_norm_layer(name, **kwargs):
    if name not in norm_layers:
        err_str = '"%s" is not among the following norm layer list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(models)))
        raise ValueError(err_str)
    norm_layer = norm_layers[name]
    return norm_layer

def get_model_instances(name, **kwargs):
    name = name.lower()
    if name not in models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(models)))
        raise ValueError(err_str)
    net = gluoncv.model_zoo._models[name](**kwargs)
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
