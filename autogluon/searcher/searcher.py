import json
import pickle
import copy
import logging
from collections import OrderedDict

__all__ = ['BaseSearcher', 'RandomSampling']

logger = logging.getLogger(__name__)

class BaseSearcher(object):
    """The config generator determines how new configurations are sampled. This can take very different levels of
    complexity, from random sampling to the construction of complex empirical prediction models for promising
    configurations.
    """
    def __init__(self, configspace):
        """Basic Searcher
        Parameters:
        -----------
        configspace: ConfigSpace.ConfigurationSpace
            The configuration space to sample from. It contains the full
            specification of the Hyperparameters with their priors
        """
        self.configspace = configspace
        self._results = OrderedDict()
        self._best_model_params = None

    def get_config(self, budget):
        """Function to sample a new configuration
        This function is called inside Hyperband to query a new configuration
        Parameters
        ----------
        budget: float
            the budget for which this configuration is scheduled
        returns: (config, info_dict)
            must return a valid configuration and a (possibly empty) info dict
        """
        raise NotImplementedError('This function needs to be overwritten in %s.'%(self.__class__.__name__))

    def update(self, config, reward, model_params=None):
        """Update the searcher with the newest metric report
        """
        #if model_params is not None and reward > self.get_best_reward():
        #    self._best_model_params = model_params
        self._results[json.dumps(config)] = reward
        logger.info('Finished Task with config: {} and reward: {}'.format(json.dumps(config), reward))

    def get_best_reward(self):
        config = max(self._results, key=self._results.get)
        return self._results[config]

    def get_best_config(self):
        config = max(self._results, key=self._results.get)
        return json.loads(config)

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' +  \
            'ConfigSpace: ' + str(self.configspace) + \
            'Results: ' + str(self._results) + \
            ')'
        return reprstr


class RandomSampling(BaseSearcher):
    """Random sampling Searcher for ConfigSpace
    """
    def get_config(self, budget=None):
        """Function to sample a new configuration
        This function is called inside Hyperband to query a new configuration
        Parameters
        ----------
        budget: float
            the budget for which this configuration is scheduled
        returns: (config, info_dict)
            must return a valid configuration and a (possibly empty) info dict
        """
        return self.configspace.sample_configuration().get_dictionary()

    def update(self, *args, **kwargs):
        """Update the searcher with the newest metric report
        """
        super(RandomSampling, self).update(*args, **kwargs)
