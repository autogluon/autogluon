import logging
import multiprocessing as mp
import pickle
from collections import OrderedDict
import numpy as np

from .bayesopt.utils.debug_log import DebugLogPrinter
from ..utils import DeprecationHelper

__all__ = ['BaseSearcher',
           'RandomSearcher',
           'RandomSampling']

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

    def __init__(self, configspace, reward_attribute=None):
        """
        :param configspace: Configuration space to sample from or search in
        :param reward_attribute: Reward attribute passed to update.
            Default: 'accuracy'

        """
        self.configspace = configspace
        self._results = OrderedDict()
        if reward_attribute is None:
            reward_attribute = 'accuracy'
        self._reward_attribute = reward_attribute

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
        from ..scheduler import FIFOScheduler
        from ..scheduler.seq_scheduler import LocalSequentialScheduler

        if isinstance(scheduler, FIFOScheduler):
            self._reward_attribute = scheduler._reward_attr
            self._constraint_attribute = scheduler._constraint_attr
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
        """Update the searcher with the newest metric report

        kwargs must include the reward (key == reward_attribute). For
        multi-fidelity schedulers (e.g., Hyperband), intermediate results are
        also reported. In this case, kwargs must also include the resource
        (key == resource_attribute).
        We can also assume that if `register_pending(config, ...)` is received,
        then later on, the searcher receives `update(config, ...)` with
        milestone as resource.

        Note that for Hyperband scheduling, update is also called for
        intermediate results. _results is updated in any case, if the new
        reward value is larger than the previously recorded one. This implies
        that the best value for a config (in _results) could be obtained for
        an intermediate resource, not the final one (virtue of early stopping).
        Full details can be reconstruction from training_history of the
        scheduler.

        """
        reward = kwargs.get(self._reward_attribute)
        assert reward is not None, \
            "Missing reward attribute '{}'".format(self._reward_attribute)
        with self.LOCK:
            # _results is updated if reward is larger than the previous entry.
            # This is the correct behaviour for multi-fidelity schedulers,
            # where update is called multiple times for a config, with
            # different resource levels.
            config_pkl = pickle.dumps(config)
            old_reward = self._results.get(config_pkl, reward)
            self._results[config_pkl] = max(reward, old_reward)

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

    def remove_case(self, config, **kwargs):
        """Remove data case previously appended by update

        For searchers which maintain the dataset of all cases (reports) passed
        to update, this method allows to remove one case from the dataset.
        """
        pass

    def evaluation_failed(self, config, **kwargs):
        """
        Called by scheduler if an evaluation job for config failed. The
        searcher should react appropriately (e.g., remove pending evaluations
        for this config, and blacklist config).
        """
        pass

    def dataset_size(self):
        """
        :return: Size of dataset a model is fitted to, or 0 if no model is
            fitted to data
        """
        return 0

    def cumulative_profile_record(self):
        """
        If profiling is supported and active, the searcher accumulates
        profiling information over get_config calls, the corresponding dict
        is returned here.
        """
        return dict()

    def model_parameters(self):
        """
        :return: Dictionary with current model (hyper)parameter values if
            this is supported; otherwise empty
        """
        return dict()

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

    def get_state(self):
        """
        Together with clone_from_state, this is needed in order to store and
        re-create the mutable state of the searcher.

        The state returned here must be pickle-able. If the searcher object is
        pickle-able, the default is returning self.

        :return: Pickle-able mutable state of searcher
        """
        return self

    def clone_from_state(self, state):
        """
        Together with get_state, this is needed in order to store and
        re-create the mutable state of the searcher.

        Given state as returned by get_state, this method combines the
        non-pickle-able part of the immutable state from self with state
        and returns the corresponding searcher clone. Afterwards, self is
        not used anymore.

        If the searcher object as such is already pickle-able, then state is
        already the new searcher object, and the default is just returning it.
        In this default, self is ignored.

        :param state: See above
        :return: New searcher object
        """
        return state

    @property
    def debug_log(self):
        """
        Some BaseSearcher subclasses support writing a debug log, using
        DebugLogPrinter. See RandomSearcher for an example.

        :return: DebugLogPrinter; or None (not supported)
        """
        return None

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


class RandomSearcher(BaseSearcher):
    """Searcher which randomly samples configurations to try next.

    Parameters
    ----------
    configspace: ConfigSpace.ConfigurationSpace
        The configuration space to sample from. It contains the full
        specification of the set of hyperparameter values (with optional prior distributions over these values).

    Examples
    --------
    By default, the searcher is created along with the scheduler. For example:

    >>> import autogluon.core as ag
    >>> @ag.args(
    ...     lr=ag.space.Real(1e-3, 1e-2, log=True))
    >>> def train_fn(args, reporter):
    ...     reporter(accuracy = args.lr ** 2)
    >>> scheduler = ag.scheduler.FIFOScheduler(
    ...     train_fn, searcher='random', num_trials=10,
    ...     reward_attr='accuracy')

    This would result in a BaseSearcher with cs = train_fn.cs. You can also
    create a RandomSearcher by hand:

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

    def __init__(self, configspace, **kwargs):
        super().__init__(
            configspace, reward_attribute=kwargs.get('reward_attribute'))
        self._first_is_default = kwargs.get('first_is_default', True)
        # We use an explicit random_state here, in order to better support
        # checkpoint and resume
        self.random_state = np.random.RandomState(
            kwargs.get('random_seed', 31415927))
        # Debug log printing (optional)
        self._debug_log = kwargs.get('debug_log')
        if self._debug_log is not None:
            if isinstance(self._debug_log, bool):
                if self._debug_log:
                    self._debug_log = DebugLogPrinter()
                else:
                    self._debug_log = None
            else:
                assert isinstance(self._debug_log, DebugLogPrinter)

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
        from ..scheduler import HyperbandScheduler

        super().configure_scheduler(scheduler)
        self._resource_attribute = None
        # If the scheduler is Hyperband, we want to know the resource
        # attribute, this is used for debug_log
        if isinstance(scheduler, HyperbandScheduler):
            self._resource_attribute = scheduler._time_attr

    def get_config(self, **kwargs):
        """Sample a new configuration at random

        Returns
        -------
        A new configuration that is valid.
        """
        self.configspace.random = self.random_state
        if self._debug_log is not None:
            self._debug_log.start_get_config('random')
        if self._first_is_default and (not self._results):
            # Try default config first
            new_config = self.configspace.get_default_configuration().get_dictionary()
        else:
            new_config = self.configspace.sample_configuration().get_dictionary()
        with self.LOCK:
            num_tries = 1
            while pickle.dumps(new_config) in self._results:
                assert num_tries <= self.MAX_RETRIES, \
                    f"Cannot find new config in BaseSearcher, even after {self.MAX_RETRIES} trials"
                new_config = self.configspace.sample_configuration().get_dictionary()
                num_tries += 1
            self._results[pickle.dumps(new_config)] = self._reward_while_pending()
        if self._debug_log is not None:
            self._debug_log.set_final_config(new_config)
            # All get_config debug log info is only written here
            self._debug_log.write_block()
        return new_config

    def update(self, config, **kwargs):
        super().update(config, **kwargs)
        if self._debug_log is not None:
            config_id = self._debug_log.config_id(config)
            reward = kwargs[self._reward_attribute]
            if self._resource_attribute is not None:
                # For HyperbandScheduler, also add the resource attribute
                resource = int(kwargs[self._resource_attribute])
                config_id = config_id + ':{}'.format(resource)
            msg = "Update for config_id {}: reward = {}".format(
                config_id, reward)
            logger.info(msg)

    def get_state(self):
        state = {
            'random_state': self.random_state,
            'results': self._results}
        if self._debug_log is not None:
            state['debug_log'] = self._debug_log.get_mutable_state()
        return state

    def clone_from_state(self, state):
        new_searcher = RandomSearcher(
            self.configspace, reward_attribute=self._reward_attribute,
            first_is_default=self._first_is_default,
            debug_log=self._debug_log)
        new_searcher.random_state = state['random_state']
        new_searcher._results = state['results']
        if self._debug_log and 'debug_log' in state:
            new_searcher._debug_log.set_mutable_state(state['debug_log'])
        return new_searcher

    @property
    def debug_log(self):
        return self._debug_log


RandomSampling = DeprecationHelper(RandomSearcher, 'RandomSampling')
