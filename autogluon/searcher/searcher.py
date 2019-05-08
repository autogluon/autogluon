import logging

__all__ = ['BaseConfigGenerator', 'RandomSampling']

logger = logging.getLogger(__name__)

class BaseConfigGenerator(object):
    """The config generator determines how new configurations are sampled. This can take very different levels of
    complexity, from random sampling to the construction of complex empirical prediction models for promising
    configurations.
    """
    def __init__(self, configspace):
        """
        Parameters:
        -----------
        configspace: ConfigSpace.ConfigurationSpace
            The configuration space to sample from. It contains the full
            specification of the Hyperparameters with their priors
        """
        self.configspace = configspace

    def get_config(self, budget):
        """
        function to sample a new configuration
        This function is called inside Hyperband to query a new configuration
        Parameters
        ----------
        budget: float
            the budget for which this configuration is scheduled
        returns: (config, info_dict)
            must return a valid configuration and a (possibly empty) info dict
        """
        raise NotImplementedError('This function needs to be overwritten in %s.'%(self.__class__.__name__))

    def update(self, *kwargs):
        """Update the searcher with the newest metric report
        """
        raise NotImplementedError('This function needs to be overwritten in %s.'%(self.__class__.__name__))
        

class RandomSampling(BaseConfigGenerator):
    """
        class to implement random sampling from a ConfigSpace
    """
    def get_config(self, budget=None):
        return self.configspace.sample_configuration().get_dictionary()

    def update(self, **kwargs):
        return
