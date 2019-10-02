import os
import json
import pickle
import copy
import logging
from collections import OrderedDict
import multiprocessing as mp

from ..utils import load, DeprecationHelper

__all__ = ['BaseSearcher', 'RandomSearcher', 'RandomSampling']

logger = logging.getLogger(__name__)

class BaseSearcher(object):
    """Base Searcher (A virtual class to inherit from)

    Args:
        configspace: ConfigSpace.ConfigurationSpace
            The configuration space to sample from. It contains the full
            specification of the Hyperparameters with their priors
    """
    LOCK = mp.Lock()
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
        with self.LOCK:
            self._results[pickle.dumps(config)] = reward
        logger.info('Finished Task with config: {} and reward: {}'.format(config, reward))

    def get_best_reward(self):
        with self.LOCK:
            if len(self._results) > 0:
                config = max(self._results, key=self._results.get)
                return self._results[config]
        return 0.0

    def get_reward(self, config):
        k = pickle.dumps(config)
        with self.LOCK:
            assert k in self._results
            return self._results[k]

    def get_best_config(self):
        with self.LOCK:
            if len(self._results) > 0:
                config = max(self._results, key=self._results.get)
                return pickle.loads(config)
            else:
                return {}

    def is_best(self, config):
        best_config = max(self._results, key=self._results.get)
        return pickle.dumps(config) == best_config

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
            '\nConfigSpace: {}.'.format(str(self.configspace)) + \
            '\nNumber of Trials: {}.'.format(len(self._results)) + \
            '\nBest Config: {}'.format(self.get_best_config()) + \
            '\nBest Reward: {}'.format(self.get_best_reward()) + \
            ')'
        return reprstr


class RandomSearcher(BaseSearcher):
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
        >>> searcher = RandomSearcher(cs)
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
        while pickle.dumps(new_config) in self._results.keys():
            new_config = self.configspace.sample_configuration().get_dictionary()
        self._results[pickle.dumps(new_config)] = 0
        return new_config

    def update(self, *args, **kwargs):
        """Update the searcher with the newest metric report
        """
        super(RandomSearcher, self).update(*args, **kwargs)

RandomSampling = DeprecationHelper(RandomSearcher, 'RandomSampling')

