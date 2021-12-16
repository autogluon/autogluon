import logging
import pickle
from collections import OrderedDict

from ..space import Categorical, Space

__all__ = ['LocalSearcher']

logger = logging.getLogger(__name__)


class LocalSearcher(object):
    """Local Searcher (virtual class to inherit from if you are creating a custom Searcher).

    Parameters
    ----------
    search_space: dict
        The configuration space to sample from. It contains the full
        specification of the Hyperparameters with their priors
    """
    def __init__(self, search_space, reward_attribute=None, **kwargs):
        """
        :param search_space: Configuration space to sample from or search in
        :param reward_attribute: Reward attribute passed to update.
            Default: 'accuracy'

        """
        self.search_space = search_space
        self._results = OrderedDict()
        if reward_attribute is None:
            reward_attribute = 'accuracy'
        self._reward_attribute = reward_attribute
        self._params_default = self._get_params_default()

    def _get_params_default(self) -> dict:
        params_default = dict()
        for key, val in self.search_space.items():
            if isinstance(val, Space):
                if isinstance(val, Categorical):
                    # FIXME: Don't do this, fix the outer code to not require this
                    d = val.data[0]
                else:
                    d = val.default
                params_default[key] = d
        return params_default


    # FIXME: Consider removing
    def configure_scheduler(self, scheduler):
        """
        Some searchers need to obtain information from the scheduler they are
        used with, in order to configure themselves.
        This method has to be called before the searcher can be used.

        The implementation here sets _reward_attribute for schedulers which
        specify it.

        Args:
            scheduler: TaskScheduler
                Scheduler the searcher is used with.

        """
        from ..scheduler.seq_scheduler import LocalSequentialScheduler
        if isinstance(scheduler, LocalSequentialScheduler):
            self._reward_attribute = scheduler._reward_attr

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

    def update(self, config, **kwargs):
        """Update the searcher with the newest metric report"""
        reward = kwargs.get(self._reward_attribute, None)
        assert reward is not None, "Missing reward attribute '{}'".format(self._reward_attribute)
        config_pkl = pickle.dumps(config)
        self._results[config_pkl] = reward

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

    def evaluation_failed(self, config, **kwargs):
        """
        Called by scheduler if an evaluation job for config failed. The
        searcher should react appropriately (e.g., remove pending evaluations
        for this config, and blacklist config).
        """
        pass

    def get_best_reward(self):
        """Calculates the reward (i.e. validation performance) produced by training under the best configuration identified so far.
           Assumes higher reward values indicate better performance.
        """
        if self._results:
            return max(self._results.values())
        return self._reward_while_pending()

    def get_reward(self, config):
        """Calculates the reward (i.e. validation performance) produced by training with the given configuration.
        """
        k = pickle.dumps(config)
        assert k in self._results
        return self._results[k]

    def get_best_config(self):
        """Returns the best configuration found so far.
        """
        if self._results:
            config_pkl = max(self._results, key=self._results.get)
            return pickle.loads(config_pkl)
        else:
            return dict()
