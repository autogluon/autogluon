from .searcher import BaseSearcher
from sklearn.model_selection import ParameterGrid

__all__ = ['GridSearcher']


class GridSearcher(BaseSearcher):
    """Grid Searcher that exhaustively tries all possible configurations. 
       This Searcher can only be used for discrete search spaces of type :class:`autogluon.space.Categorical`

    Examples
    --------
    >>> import autogluon as ag
    >>> @ag.args(
    ...     x=ag.space.Categorical(0, 1, 2),
    ...     y=ag.space.Categorical('a', 'b', 'c'))
    >>> def train_fn(args, reporter):
    ...     pass
    >>> searcher = ag.searcher.GridSearcher(train_fn.cs)
    >>> searcher.get_config()
    Number of configurations for grid search is 9
    {'x.choice': 2, 'y.choice': 2}
    """
    def __init__(self, configspace, **kwargs):
        super().__init__(
            configspace, reward_attribute=kwargs.get('reward_attribute'))
        param_grid = {}
        hp_ordering = configspace.get_hyperparameter_names()
        for hp in hp_ordering:
            hp_obj = configspace.get_hyperparameter(hp)
            hp_type = str(type(hp_obj)).lower()
            assert 'categorical' in hp_type, \
                'Only Categorical is supported, but {} is {}'.format(hp, hp_type)
            param_grid[hp] = hp_obj.choices

        self._configs = list(ParameterGrid(param_grid))
        print('Number of configurations for grid search is {}'.format(len(self._configs)))

    def __len__(self):
        return len(self._configs)

    def get_config(self):
        """ Return new hyperparameter configuration to try next.
        """
        return self._configs.pop()
