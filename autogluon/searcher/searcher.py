import logging
import multiprocessing as mp
import pickle
from collections import OrderedDict

from ..utils import DeprecationHelper

__all__ = ['BaseSearcher', 'RandomSearcher', 'RandomSampling']

logger = logging.getLogger(__name__)


class BaseSearcher(object):
    """Base Searcher (virtual class to inherit from if you are creating a custom Searcher).

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

    @staticmethod
    def _reward_while_pending():
        """Defines the reward value which is assigned to config, while it is pending."""
        return float("-inf")

    def get_config(self, **kwargs):
        """Function to sample a new configuration

        This function is called inside TaskScheduler to query a new configuration

        Args:
        kwargs:
            Extra information may be passed from scheduler to searcher
        returns: (config, info_dict)
            must return a valid configuration and a (possibly empty) info dict
        """
        raise NotImplementedError(f'This function needs to be overwritten in {self.__class__.__name__}.')

    def update(self, config, reward, **kwargs):
        """Update the searcher with the newest metric report

        Note that for multi-fidelity schedulers (e.g., Hyperband), 
        intermediate results are also reported. In this case, the time attribute is
        among `**kwargs`. We can also assume that if
        `register_pending(config, ...)` is received, then later on,
        the searcher receives `update(config, ...)` with milestone as time attribute.
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
            logger.info(f'Finished Task with config: {config} and reward: {reward}')

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
        """Calculates the reward (i.e. validation performance) produced by training under the best configuration identified so far.
           Assumes higher reward values indicate better performance.
        """
        with self.LOCK:
            if self._results:
                return max(self._results.values())
        return self._reward_while_pending()

    def get_reward(self, config):
        """Calculates the reward (i.e. validation performance) produced by training with the given configuration.
        """
        k = pickle.dumps(config)
        with self.LOCK:
            assert k in self._results
            return self._results[k]

    def get_best_config(self):
        """Returns the best configuration found so far.
        """
        with self.LOCK:
            if self._results:
                config_pkl = max(self._results, key=self._results.get)
                return pickle.loads(config_pkl)
            else:
                return dict()

    def get_best_config_reward(self):
        """Returns the best configuration found so far, as well as the reward associated with this best config.
        """
        with self.LOCK:
            if self._results:
                config_pkl = max(self._results, key=self._results.get)
                return pickle.loads(config_pkl), self._results[config_pkl]
            else:
                return dict(), self._reward_while_pending()

    def __repr__(self):
        config, reward = self.get_best_config_reward()
        reprstr = (
                f'{self.__class__.__name__}(' +
                f'\nConfigSpace: {self.configspace}.' +
                f'\nNumber of Trials: {len(self._results)}.' +
                f'\nBest Config: {config}' +
                f'\nBest Reward: {reward}' +
                f')'
        )
        return reprstr


# TODO: Does not use default hyperparams for first run
class RandomSearcher(BaseSearcher):
    """Searcher which randomly samples configurations to try next.

    Parameters
    ----------
    configspace: ConfigSpace.ConfigurationSpace
        The configuration space to sample from. It contains the full
        specification of the set of hyperparameter values (with optional prior distributions over these values).

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
    MAX_RETRIES = 100

    def get_config(self, **kwargs):
        """Sample a new configuration at random

        Returns
        -------
        A new configuration that is valid.
        """
        if not self._results:  # no hyperparams have been tried yet, first try default config
            new_config = self.configspace.get_default_configuration().get_dictionary()
        else:
            new_config = self.configspace.sample_configuration().get_dictionary()
        with self.LOCK:
            num_tries = 1
            while pickle.dumps(new_config) in self._results.keys():
                assert num_tries <= self.MAX_RETRIES, \
                    f"Cannot find new config in BaseSearcher, even after {self.MAX_RETRIES} trials"
                new_config = self.configspace.sample_configuration().get_dictionary()
                num_tries += 1
            self._results[pickle.dumps(new_config)] = self._reward_while_pending()
        return new_config


RandomSampling = DeprecationHelper(RandomSearcher, 'RandomSampling')
