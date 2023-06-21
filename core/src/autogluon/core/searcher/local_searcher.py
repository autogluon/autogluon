import logging
import pickle
from collections import OrderedDict

from autogluon.common import space

__all__ = ["LocalSearcher"]

logger = logging.getLogger(__name__)


class LocalSearcher(object):
    """Local Searcher (virtual class to inherit from if you are creating a custom Searcher).

    Parameters
    ----------
    search_space: dict
        The configuration space to sample from. It contains the full
        specification of the Hyperparameters with their priors
    """

    def __init__(self, search_space: dict, reward_attribute: str = "reward", **kwargs):
        """
        :param search_space: Configuration space to sample from or search in
        :param reward_attribute: Reward attribute passed to update.
            Default: 'reward'

        """
        self.search_space = search_space
        self._results = OrderedDict()
        self._reward_attribute = reward_attribute
        self._params_static = self._get_params_static()
        self._params_default = self._get_params_default(self._params_static)
        self._params_order = list(self._params_default.keys())
        self._params_cat_dict = self._get_params_cat_dict()

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
        raise NotImplementedError(f"This function needs to be overwritten in {self.__class__.__name__}.")

    def update(self, config: dict, **kwargs):
        """
        Update the searcher with the newest metric report.
        Will error if config contains unknown parameters, values outside the valid search space, or is missing parameters.
        """
        reward = kwargs.get(self._reward_attribute, None)
        assert reward is not None, "Missing reward attribute '{}'".format(self._reward_attribute)
        self._add_result(config=config, result=reward)

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
        """Calculates the reward (i.e. validation performance) produced by training with the given configuration."""
        config_pkl = self._pickle_config(config=config)
        assert config_pkl in self._results
        return self._results[config_pkl]

    def get_best_config(self):
        """Returns the best configuration found so far."""
        if self._results:
            config_pkl = max(self._results, key=self._results.get)
            return self._unpickle_config(config_pkl=config_pkl)
        else:
            return dict()

    def get_results(self, sort=True) -> list:
        """
        Gets a list of results in the form (config, reward).

        Parameters
        ----------
        sort : bool, default = True
            If True, sorts the configs in order from best to worst reward.
            If False, config order is undefined.
        """
        results = []
        for config_pkl, reward in self._results.items():
            config = self._unpickle_config(config_pkl=config_pkl)
            results.append((config, reward))
        if sort:
            results = sorted(results, key=lambda x: x[1], reverse=True)
        return results

    def _get_params_static(self) -> dict:
        """
        Gets a dictionary of static key values, where no search space is used and therefore the values are always the same in all configs.
        """
        params_static = dict()
        for key, val in self.search_space.items():
            if not isinstance(val, space.Space):
                params_static[key] = val
        return params_static

    def _get_params_default(self, params_static: dict) -> dict:
        """
        Gets the default config by calling `val.default` on every search space parameter, plus the static key values.
        """
        params_default = dict()
        for key, val in self.search_space.items():
            if isinstance(val, space.Space):
                params_default[key] = val.default
        params_default.update(params_static)
        return params_default

    def _get_params_cat_dict(self) -> dict:
        """
        Gets the dictionary of pickled category value -> index mapping for Category search spaces.
        This is used in `self._pickle_config` to map values to idx when pickling the config. This compresses the size of the pkl file.
        When being later unpickled via `self._unpickle_config`, the idx can be used to get the key value via `self.search_space[key][idx]`.
        """
        params_cat_dict = dict()
        for key, val in self.search_space.items():
            if isinstance(val, space.Categorical):
                cat_map = dict()
                for i, cat in enumerate(val.data):
                    cat_pkl = pickle.dumps(cat)
                    cat_map[cat_pkl] = i

                params_cat_dict[key] = cat_map
        return params_cat_dict

    def _add_result(self, config: dict, result: float):
        assert isinstance(result, (float, int)), f"result must be a float or int! Was instead {type(result)} | Value: {result}"
        config_pkl = self._pickle_config(config=config)
        self._results[config_pkl] = result

    def _pickle_config(self, config: dict) -> bytes:
        assert isinstance(config, dict), f"config must be a dict! Was instead {type(config)} | Value: {config}"
        assert len(config) == len(self._params_order), (
            f"Config length does not match expected params count!\n" f"Expected: {self._params_order}\n" f"Actual:   {list(config.keys())}"
        )

        # Note: This code is commented out because it can be computationally and memory expensive if user sends large objects in search space, such as datasets.
        """
        for key in self._params_static:
            assert pickle.dumps(config[key]) == pickle.dumps(self._params_static[key]), \
                f'Invalid config value for search space parameter "{key}" | Invalid Value: {config[key]} | Expected Value: {self._params_static[key]}'
        """
        config_to_pkl = []
        for key in self._params_order:
            if key in self._params_static:
                pass
            elif key in self._params_cat_dict:
                try:
                    cat_idx = self._params_cat_dict[key][pickle.dumps(config[key])]
                except KeyError:
                    raise AssertionError(
                        f'Invalid config value for search space parameter "{key}" | '
                        f"Invalid Value: {config[key]} | Valid Values: {self.search_space[key].data}"
                    )
                config_to_pkl.append(cat_idx)
            else:
                config_to_pkl.append(config[key])
        return pickle.dumps(config_to_pkl)

    def _unpickle_config(self, config_pkl: bytes) -> dict:
        assert isinstance(config_pkl, bytes), f"config_pkl must be a bytes object! Was instead {type(config_pkl)} | Value: {config_pkl}"
        config_compressed = pickle.loads(config_pkl)
        config = dict()
        i = -1
        for key in self._params_order:
            if key in self._params_static:
                config[key] = self._params_static[key]
            else:
                i += 1
                val = config_compressed[i]
                if key in self._params_cat_dict:
                    config[key] = self.search_space[key][val]
                else:
                    config[key] = val
        return config
