from autogluon.common import space
from autogluon.core.searcher import LocalRandomSearcher


def test_local_random_searcher():
    search_space = dict(
        a=space.Real(0, 1, default=0.2),
        b=space.Real(0.05, 1, default=0.4, log=True),
        c=space.Int(5, 15),
        d=space.Int(7, 23, default=16),
        e=space.Categorical("a", 7, ["hello", 2]),
    )

    searcher = LocalRandomSearcher(search_space=search_space)

    expected_config_1 = {"a": 0.2, "b": 0.4, "c": 5, "d": 16, "e": "a"}
    expected_config_2 = {"a": 0.5488135039273248, "b": 0.4260424000595025, "c": 8, "d": 10, "e": 7}
    expected_config_3 = {"a": 0.6235636967859723, "b": 0.15814742875130683, "c": 12, "d": 13, "e": "a"}
    expected_config_4 = {"a": 0.9636627605010293, "b": 0.1577026248478398, "c": 11, "d": 14, "e": ["hello", 2]}
    expected_config_5 = {"a": 0.5680445610939323, "b": 0.800200824711684, "c": 13, "d": 16, "e": "a"}

    assert searcher.get_best_reward() == float("-inf")
    config1 = searcher.get_config()
    assert searcher.get_reward(config1) == float("-inf")
    assert searcher.get_best_reward() == float("-inf")
    searcher.update(config1, reward=0.2)
    assert searcher.get_reward(config1) == 0.2
    assert searcher.get_best_reward() == 0.2
    assert searcher.get_best_config() == config1

    config2 = searcher.get_config()

    config3 = searcher.get_config()

    config4 = searcher.get_config()
    searcher.update(config4, reward=0.1)
    assert searcher.get_reward(config4) == 0.1
    assert searcher.get_best_reward() == 0.2
    assert searcher.get_best_config() == config1
    searcher.update(config4, reward=0.5)
    assert searcher.get_reward(config4) == 0.5
    assert searcher.get_best_reward() == 0.5
    assert searcher.get_best_config() == config4

    config5 = searcher.get_config()

    assert expected_config_1 == config1
    assert expected_config_2 == config2
    assert expected_config_3 == config3
    assert expected_config_4 == config4
    assert expected_config_5 == config5

    assert len(searcher._results) == 5
