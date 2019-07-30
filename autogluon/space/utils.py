import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

__all__ = ['sample_configuration']

def sample_configuration(spaces):
    cs = CS.ConfigurationSpace()
    hparams = []
    for space in spaces:
        hparams.append(space.hyper_param)
    cs.add_hyperparameters(hparams)
    return cs.sample_configuration()
