import os
import json
import pickle
import copy
import logging
from collections import OrderedDict

from ..basic import load

__all__ = ['BaseSearcher', 'RandomSampling']

logger = logging.getLogger(__name__)

class BaseSearcher(object):
    """Base Searcher (A virtual class to inherit from)

    Args:
        configspace: ConfigSpace.ConfigurationSpace
            The configuration space to sample from. It contains the full
            specification of the Hyperparameters with their priors
    """
    def __init__(self, configspace):
        self.configspace = configspace
        self._results = OrderedDict()
        self._best_state_path = None

    def get_config(self):
        """Function to sample a new configuration

        This function is called inside TaskScheduler to query a new configuration

        Args:
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

    def is_best(self, config):
        best_config = max(self._results, key=self._results.get)
        return json.dumps(config) == best_config

    def get_best_state_path(self):
        assert os.path.isfile(self._best_state_path), \
            'Please use report_best_state_pather.save_dict(model_params) during the training.'
        return self._best_state_path

    def get_best_state(self):
        assert os.path.isfile(self._best_state_path), \
            'Please use report_best_state_pather.save_dict(model_params) during the training.'
        return load(self._best_state_path)

    def update_best_state(self, filepath):
        self._best_state_path = filepath

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' +  \
            'ConfigSpace: ' + str(self.configspace) + \
            'Results: ' + str(self._results) + \
            ')'
        return reprstr


class RandomSampling(BaseSearcher):
    """Random sampling Searcher for ConfigSpace

    Args:
        configspace: ConfigSpace.ConfigurationSpace
            The configuration space to sample from. It contains the full
            specification of the Hyperparameters with their priors

    Example:
        >>> import ConfigSpace as CS
        >>> import ConfigSpace.hyperparameters as CSH
        >>> # create configuration space
        >>> cs = CS.ConfigurationSpace()
        >>> lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1e-1, log=True)
        >>> cs.add_hyperparameter(lr)
        >>> # create searcher
        >>> searcher = RandomSampling(cs)
        >>> searcher.get_config()
    """
    def get_config(self):
        """Function to sample a new configuration
        This function is called inside Hyperband to query a new configuration

        Args:
            returns: (config, info_dict)
                must return a valid configuration and a (possibly empty) info dict
        """
        new_config = self.configspace.sample_configuration().get_dictionary()
        while json.dumps(new_config) in self._results.keys():
            new_config = self.configspace.sample_configuration().get_dictionary()
        self._results[json.dumps(new_config)] = 0
        return new_config

    def update(self, *args, **kwargs):
        """Update the searcher with the newest metric report
        """
        super(RandomSampling, self).update(*args, **kwargs)
