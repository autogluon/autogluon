import gluoncv as gcv
from .models.efficientnet import *
from .models.standford_dog_models import *

_all__ = ['get_model', 'get_model_list']

models = {
    'efficientnet_b0': get_efficientnet_b0,
    'efficientnet_b1': get_efficientnet_b1,
    'efficientnet_b2': get_efficientnet_b2,
    'efficientnet_b3': get_efficientnet_b3,
    'efficientnet_b4': get_efficientnet_b4,
    'efficientnet_b5': get_efficientnet_b5,
    'efficientnet_b6': get_efficientnet_b6,
    'efficientnet_b7': get_efficientnet_b7,
    'standford_dog_resnet152_v1': standford_dog_resnet152_v1,
    'standford_dog_resnext101_64x4d': standford_dog_resnext101_64x4d,
    }


def get_model(name, **kwargs):
    """Returns a pre-defined model by name
    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Returns
    -------
    Module:
        The model.
    """

    name = name.lower()
    if name in models:
        net = models[name](**kwargs)
    elif name in gcv.model_zoo.get_model_list():
        net = gcv.model_zoo.get_model(name, **kwargs)
    else:
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models.keys()))))
    return net


def get_model_list():
    """Get the entire list of model names in model_zoo.
    Returns
    -------
    list of str
        Entire list of model names in model_zoo.
    """
    return list(models.keys()) + list(gcv.model_zoo.get_model_list())
