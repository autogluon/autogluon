from .models.efficientnet import *
from .models.standford_dog_models import *

_all__ = ['get_model']

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
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)
    return net
