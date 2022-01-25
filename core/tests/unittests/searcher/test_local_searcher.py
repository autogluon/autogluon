
from autogluon.core.searcher.local_searcher import LocalSearcher


def test_local_searcher():
    search_space = {'hello': 'default', 7: 42}
    searcher = LocalSearcher(search_space=search_space)

    config1 = {'hello': 'world', 7: 'str'}
    config2 = {'hello': 'test', 7: None}

    assert searcher.get_best_reward() == float("-inf")
    searcher.update(config1, accuracy=0.2)
    assert searcher.get_best_reward() == 0.2
    assert searcher.get_best_config() == config1

    searcher.update(config1, accuracy=0.1)
    assert searcher.get_best_reward() == 0.1
    assert searcher.get_best_config() == config1

    searcher.update(config2, accuracy=0.7)
    assert searcher.get_best_reward() == 0.7
    assert searcher.get_best_config() == config2


def test_local_searcher_pickle():
    search_space = {1: 'default', 2: 42}
    searcher = LocalSearcher(search_space=search_space)

    # Identical configs should have same pkl key, different configs should have different pkl keys
    config_1 = {1: 1, 2: 2}
    config_2 = {2: 2, 1: 1}
    config_diff = {1: 2, 2: 1}
    config_pkl_1 = searcher._pickle_config(config_1)
    config_pkl_2 = searcher._pickle_config(config_2)
    config_pkl_diff = searcher._pickle_config(config_diff)

    assert config_pkl_1 == config_pkl_2
    assert config_pkl_1 != config_pkl_diff

    config_unpkl = searcher._unpickle_config(config_pkl_1)
    config_unpkl_diff = searcher._unpickle_config(config_pkl_diff)

    assert config_unpkl == config_1
    assert config_unpkl == config_2
    assert config_unpkl_diff == config_diff
