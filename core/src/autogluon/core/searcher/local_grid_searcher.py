import logging
from typing import Dict

import numpy as np
from sklearn.model_selection import ParameterGrid

from autogluon.common import space as ag_space

from .local_searcher import LocalSearcher

__all__ = ["LocalGridSearcher"]

logger = logging.getLogger(__name__)


class LocalGridSearcher(LocalSearcher):
    """
    Grid Searcher that exhaustively tries all possible configurations. Grid with Int/Real ignores 'default' values in search spaces.
    """

    def __init__(self, grid_numeric_spaces_points_number=4, grid_num_sample_settings: Dict = None, **kwargs):
        """
        Parameters
        ----------
        grid_numeric_spaces_points_number: int, default = 4
            number of data point to sample from numeric space
        grid_num_sample_settings: dict (optional), default = None
            mapping between numeric space name and number of points to sample from it. Example {'a': 4} means
            sample 4 points from space 'a'. If no value present in this map, then `grid_numeric_spaces_points_number` will be used.
        """
        super().__init__(**kwargs)
        self._grid_numeric_spaces_points_number = grid_numeric_spaces_points_number
        self._grid_num_sample_settings = grid_num_sample_settings
        self._params_space = self._get_params_space()
        self._params_grid = ParameterGrid(self._params_space)
        self._grid_index = 0
        self._grid_length = len(self._params_grid)

    def _get_params_space(self) -> dict:
        param_space = dict()
        for key, val in self.search_space.items():
            if isinstance(val, ag_space.Space):
                samples_num = self._get_samples_number(key)
                if isinstance(val, ag_space.Int):
                    samples = min(val.upper - val.lower + 1, samples_num)
                    param_space[key] = np.linspace(val.lower, val.upper, samples, dtype=int)
                elif isinstance(val, ag_space.Real):
                    space = np.geomspace if val.log else np.linspace
                    param_space[key] = space(val.lower, val.upper, num=samples_num)
                elif isinstance(val, ag_space.Categorical):
                    sk = val.convert_to_sklearn()
                    param_space[key] = sk
                else:
                    raise AssertionError(f'Only Categorical is supported, but parameter "{key}" is type: {type(val)}')

        return param_space

    def _get_samples_number(self, key):
        samples = self._grid_numeric_spaces_points_number
        if self._grid_num_sample_settings is not None:
            samples = self._grid_num_sample_settings.get(key, samples)
        return samples

    def __len__(self):
        return self._grid_length - self._grid_index

    def get_config(self):
        """Return new hyperparameter configuration to try next."""
        if len(self) <= 0:
            raise AssertionError(
                f"No configs left to get. All {self._grid_length} configs have been accessed already."
            )
        config = self._params_grid[self._grid_index]
        self._grid_index += 1
        for key, val in config.items():
            # If this isn't done, warnings are spammed in XGBoost
            if isinstance(val, np.int64):
                config[key] = int(val)
            elif isinstance(val, np.float64):
                config[key] = float(val)
        return config
