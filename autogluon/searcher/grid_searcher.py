__all__ = ['GridSearcher']

from .searcher import BaseSearcher
from ..core.space import Categorical

class GridSearcher(BaseSearcher):
    """Grid Searcher
    """
    def __init__(self, cs):
        super().__init__(cs)
        from sklearn.model_selection import ParameterGrid
        param_grid = {}
        hp_ordering = cs.get_hyperparameter_names()
        for hp in hp_ordering:
            hp_obj = cs.get_hyperparameter(hp)
            hp_type = str(type(hp_obj)).lower()
            assert 'categorical' in hp_type, \
                'Only Categorical is supported, but {} is {}'.format(hp, hp_type)
            param_grid[hp] = hp_obj.choices

        self._configs = list(ParameterGrid(param_grid))
        print('Number of configurations for grid search is {}'.format(len(self._configs)))

    def __len__(self):
        return len(self._configs)

    def get_config(self):
        return self._configs.pop()
