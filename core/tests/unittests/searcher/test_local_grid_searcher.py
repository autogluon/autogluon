from autogluon.core.space import Categorical, Int, Real
from autogluon.core.searcher import LocalGridSearcher


def test_local_grid_searcher():
    search_space = dict(
        a=Categorical('a', 7, ['hello', 2]),
        b=Categorical(12, 15),
    )

    searcher = LocalGridSearcher(search_space=search_space)

    expected_config_1 = {'b': 12, 'a': 'a'}
    expected_config_2 = {'b': 15, 'a': 'a'}
    expected_config_3 = {'b': 12, 'a': 7}
    expected_config_4 = {'b': 15, 'a': 7}
    expected_config_5 = {'b': 12, 'a': ['hello', 2]}
    expected_config_6 = {'b': 15, 'a': ['hello', 2]}

    config1 = searcher.get_config()
    searcher.update(config1, accuracy=0.2)
    assert searcher.get_reward(config1) == 0.2
    assert searcher.get_best_reward() == 0.2
    assert searcher.get_best_config() == config1

    config2 = searcher.get_config()

    config3 = searcher.get_config()

    config4 = searcher.get_config()
    searcher.update(config4, accuracy=0.1)
    assert searcher.get_reward(config4) == 0.1
    assert searcher.get_best_reward() == 0.2
    assert searcher.get_best_config() == config1
    searcher.update(config4, accuracy=0.5)
    assert searcher.get_reward(config4) == 0.5
    assert searcher.get_best_reward() == 0.5
    assert searcher.get_best_config() == config4

    config5 = searcher.get_config()

    config6 = searcher.get_config()

    assert expected_config_1 == config1
    assert expected_config_2 == config2
    assert expected_config_3 == config3
    assert expected_config_4 == config4
    assert expected_config_5 == config5
    assert expected_config_6 == config6

    assert len(searcher._results) == 2

    try:
        searcher.get_config()
    except AssertionError:
        pass
    else:
        raise AssertionError('GridSearcher should error due to being out of configs')


def test_invalid_local_grid_searcher():
    search_spaces = [
        dict(a=Int(12, 15)),
        dict(a=Real(12, 15)),
    ]

    for search_space in search_spaces:
        try:
            LocalGridSearcher(search_space=search_space).get_config()
        except AssertionError:
            pass
        else:
            raise AssertionError(f'GridSearcher should error due to invalid search space types. search_space: {search_space}')
