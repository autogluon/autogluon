import os
import pickle
import logging
from collections import OrderedDict
import multiprocessing as mp

from ..utils import load, DeprecationHelper

__all__ = ['BaseSearcher', 'RandomSearcher', 'RandomSampling']

logger = logging.getLogger(__name__)

class BaseSearcher(object):
    """Base Searcher (A virtual class to inherit from)

    Parameters
    ----------
    configspace: ConfigSpace.ConfigurationSpace
        The configuration space to sample from. It contains the full
        specification of the Hyperparameters with their priors
    """
    LOCK = mp.Lock()
    def __init__(self, configspace):
        self.configspace = configspace
        self._results = OrderedDict()
        self._best_state_path = None

    def get_config(self, **kwargs):
        """Function to sample a new configuration

        This function is called inside TaskScheduler to query a new configuration

        Args:
        kwargs:
            Extra information may be passed from scheduler to searcher
        returns: (config, info_dict)
            must return a valid configuration and a (possibly empty) info dict
        """
        raise NotImplementedError('This function needs to be overwritten in %s.'%(self.__class__.__name__))

    def update(self, config, reward, **kwargs):
        """Update the searcher with the newest metric report

        Note that for multi-fidelity schedulers (e.g., Hyperband), also
        intermediate results are reported. In this case, the time attribute is
        among **kwargs. We can also assume that if
        register_pending(config, ...) is received, then later on,
        the searcher receives update(config, ...) with milestone as time attribute.
        """
        is_done = kwargs.get('done', False)
        is_terminated = kwargs.get('terminated', False)
        # Only if evaluation is done or terminated (otherwise, it is an intermediate
        # result)
        if is_done or is_terminated:
            with self.LOCK:
                # Note: In certain versions of a scheduler, we may see 'terminated'
                # several times for the same config. In this case, we log the best
                # (largest) result here
                config_pkl = pickle.dumps(config)
                old_reward = self._results.get(config_pkl, reward)
                self._results[config_pkl] = max(reward, old_reward)
            logger.info('Finished Task with config: {} and reward: {}'.format(
                config, reward))

    def register_pending(self, config, milestone=None):
        """
        Signals to searcher that evaluation for config has started, but not
        yet finished, which allows model-based searchers to register this
        evaluation as pending.
        For multi-fidelity schedulers, milestone is the next milestone the
        evaluation will attend, so that model registers (config, milestone)
        as pending.
        In general, the searcher may assume that update is called with that
        config at a later time.
        """
        pass

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

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' +  \
            '\nConfigSpace: {}.'.format(str(self.configspace)) + \
            '\nNumber of Trials: {}.'.format(len(self._results)) + \
            '\nBest Config: {}'.format(self.get_best_config()) + \
            '\nBest Reward: {}'.format(self.get_best_reward()) + \
            ')'
        return reprstr


# TODO: Does not use default hyperparams for first run
class RandomSearcher(BaseSearcher):
    """Random sampling Searcher for ConfigSpace

    Parameters
    ----------
    configspace: ConfigSpace.ConfigurationSpace
        The configuration space to sample from. It contains the full
        specification of the Hyperparameters with their priors

    Examples
    --------
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
    def get_config(self, **kwargs):
        """Function to sample a new configuration
        This function is called inside Hyperband to query a new configuration

        Parameters
        ----------
        returns: (config, info_dict)
            must return a valid configuration and a (possibly empty) info dict
        """
        if len(self._results) == 0: # no hyperparams have been tried yet, first try default config
            new_config = self.configspace.get_default_configuration().get_dictionary()
        else:
            new_config = self.configspace.sample_configuration().get_dictionary()
        while pickle.dumps(new_config) in self._results.keys(): # TODO: may never terminate
            new_config = self.configspace.sample_configuration().get_dictionary()
        self._results[pickle.dumps(new_config)] = 0
        return new_config

    def update(self, config, reward, **kwargs):
        """Update the searcher with the newest metric report
        """
        super(RandomSearcher, self).update(config, reward, **kwargs)


RandomSampling = DeprecationHelper(RandomSearcher, 'RandomSampling')
