from .models.efficientnet import *

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
        }
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)
    return net
