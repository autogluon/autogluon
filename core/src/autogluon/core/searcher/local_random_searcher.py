import logging
import pickle

import numpy as np
from sklearn.model_selection import ParameterSampler

from .local_searcher import LocalSearcher
from ..space import Space

__all__ = ['LocalRandomSearcher']

logger = logging.getLogger(__name__)


class LocalRandomSearcher(LocalSearcher):
    """Searcher which randomly samples configurations to try next."""
    MAX_RETRIES = 100

    def __init__(self, *, first_is_default=True, random_seed=0, **kwargs):
        super().__init__(**kwargs)
        self._first_is_default = first_is_default
        # We use an explicit random_state here, in order to better support checkpoint and resume
        self.random_state = np.random.RandomState(random_seed)
        self._params_space = self._get_params_space()

    def _get_params_space(self) -> dict:
        param_space = dict()
        for key, val in self.search_space.items():
            if isinstance(val, Space):
                sk = val.convert_to_sklearn()
                param_space[key] = sk
        return param_space

    def _sample_config(self) -> dict:
        params = list(ParameterSampler(self._params_space, n_iter=1, random_state=self.random_state))[0]
        for key in params:
            if isinstance(params[key], np.float64):
                # Fix error in FastAI, can't handle np.float64
                params[key] = float(params[key])
        return params

    def get_config(self, **kwargs) -> dict:
        """Sample a new configuration at random

        Returns
        -------
        A new configuration that is valid.
        """
        if self._first_is_default and (not self._results):
            # Try default config first
            new_config = self._params_default
        else:
            new_config = self._sample_config()
        num_tries = 1
        while pickle.dumps(new_config) in self._results:
            assert num_tries <= self.MAX_RETRIES, f"Cannot find new config in LocalRandomSearcher, even after {self.MAX_RETRIES} trials"
            new_config = self._sample_config()
            num_tries += 1
        self._results[pickle.dumps(new_config)] = self._reward_while_pending()
        return new_config
