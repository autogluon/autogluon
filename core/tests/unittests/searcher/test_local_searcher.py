
import unittest

from autogluon.core.searcher.local_searcher import LocalSearcher
from autogluon.core.space import Categorical


class TestLocalSearcher(unittest.TestCase):
    def test_local_searcher(self):
        search_space = {'param1': Categorical('hello', 'world'), 7: 42}
        searcher = LocalSearcher(search_space=search_space)

        config1 = {'param1': 'hello', 7: 42}
        config2 = {'param1': 'world', 7: 42}
        config_edge_case_1 = {'param1': 'world', 7: 0}
        config_invalid_2 = {'param1': 'invalid', 7: 42}
        config_invalid_3 = {'param1': 'hello'}
        config_invalid_4 = {7: 42}
        config_invalid_5 = {}
        config_invalid_6 = {'param1': 'hello', 7: 42, 'unknown_param': 7}
        config_invalid_7 = 0

        assert searcher.get_best_reward() == float("-inf")
        searcher.update(config1, reward=0.2)
        assert searcher.get_best_reward() == 0.2
        assert searcher.get_best_config() == config1

        assert searcher.get_results() == [({'param1': 'hello', 7: 42}, 0.2)]

        searcher.update(config1, reward=0.1)
        assert searcher.get_best_reward() == 0.1
        assert searcher.get_best_config() == config1

        assert searcher.get_results() == [({'param1': 'hello', 7: 42}, 0.1)]

        searcher.update(config2, reward=0.7)
        assert searcher.get_best_reward() == 0.7
        assert searcher.get_best_config() == config2

        assert searcher.get_results() == [({'param1': 'world', 7: 42}, 0.7), ({'param1': 'hello', 7: 42}, 0.1)]
        assert len(searcher._results) == 2
        # This config is technically invalid, but for performance reasons is allowed to avoid having to pickle compare static parameters.
        # Since the static parameter should be fixed, this config is treated as being equivalent to config2
        searcher.update(config_edge_case_1, reward=0)
        assert searcher.get_best_reward() == 0.1
        assert len(searcher._results) == 2
        self.assertRaises(AssertionError, searcher.update, config_invalid_2, reward=0)
        self.assertRaises(AssertionError, searcher.update, config_invalid_3, reward=0)
        self.assertRaises(AssertionError, searcher.update, config_invalid_4, reward=0)
        self.assertRaises(AssertionError, searcher.update, config_invalid_5, reward=0)
        self.assertRaises(AssertionError, searcher.update, config_invalid_6, reward=0)
        self.assertRaises(AssertionError, searcher.update, config_invalid_7, reward=0)
        self.assertRaises(AssertionError, searcher.update, config1, reward='invalid_reward')
        assert len(searcher._results) == 2
        assert searcher.get_results() == [({'param1': 'hello', 7: 42}, 0.1), ({'param1': 'world', 7: 42}, 0)]

    def test_local_searcher_pickle(self):
        search_space = {1: Categorical(1, 2), 2: Categorical(1, 2)}
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
