import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

__all__ = ['sample_configuration']

def sample_configuration(spaces):
    """sample configuration helper.

    Args:
        spaces: the search space

    Example:
        >>> sample_configuration([list_space, linear_space, log_space])
    """
    cs = CS.ConfigurationSpace()
    hparams = []
    for space in spaces:
        hparams.append(space.hyper_param)
    cs.add_hyperparameters(hparams)
    return cs.sample_configuration()
