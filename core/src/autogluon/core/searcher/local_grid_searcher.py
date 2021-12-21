import logging

from sklearn.model_selection import ParameterGrid

from .local_searcher import LocalSearcher
from ..space import Categorical, Space

__all__ = ['LocalGridSearcher']

logger = logging.getLogger(__name__)


class LocalGridSearcher(LocalSearcher):
    """
    Grid Searcher that exhaustively tries all possible configurations.
    This Searcher can only be used for discrete search spaces of type :class:`autogluon.space.Categorical`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._params_space = self._get_params_space()
        self._params_grid = ParameterGrid(self._params_space)
        self._grid_index = 0
        self._grid_length = len(self._params_grid)

    def _get_params_space(self) -> dict:
        param_space = dict()
        for key, val in self.search_space.items():
            if isinstance(val, Space):
                if not isinstance(val, Categorical):
                    raise AssertionError(f'Only Categorical is supported, but parameter "{key}" is type: {type(val)}')
                sk = val.convert_to_sklearn()
                param_space[key] = sk
        return param_space

    def __len__(self):
        return self._grid_length - self._grid_index

    def get_config(self):
        """ Return new hyperparameter configuration to try next."""
        if len(self) <= 0:
            raise AssertionError(f'No configs left to get. All {self._grid_length} configs have been accessed already.')
        config = self._params_grid[self._grid_index]
        self._grid_index += 1
        return config
