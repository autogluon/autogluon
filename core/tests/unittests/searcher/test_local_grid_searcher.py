from autogluon.common import space
from autogluon.core.searcher import LocalGridSearcher


def test_local_grid_searcher_categorical():
    search_space = dict(
        a=space.Categorical("a", 7, ["hello", 2]),
        b=space.Categorical(12, 15),
    )

    searcher = LocalGridSearcher(search_space=search_space)

    expected_config_1 = {"b": 12, "a": "a"}
    expected_config_2 = {"b": 15, "a": "a"}
    expected_config_3 = {"b": 12, "a": 7}
    expected_config_4 = {"b": 15, "a": 7}
    expected_config_5 = {"b": 12, "a": ["hello", 2]}
    expected_config_6 = {"b": 15, "a": ["hello", 2]}

    config1 = searcher.get_config()
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
        raise AssertionError("GridSearcher should error due to being out of configs")


def test_local_grid_searcher_numeric():
    search_spaces = [
        [dict(a=space.Bool()), [{"a": 0}, {"a": 1}]],
        [dict(a=space.Int(12, 15)), [{"a": 12}, {"a": 13}, {"a": 14}, {"a": 15}]],
        [dict(a=space.Real(12, 16)), [{"a": 12.0}, {"a": 13.333333333333334}, {"a": 14.666666666666666}, {"a": 16.0}]],
        [
            dict(a=space.Real(12, 16, log=True)),
            [{"a": 12.0}, {"a": 13.207708995578509}, {"a": 14.536964742657117}, {"a": 16.0}],
        ],
    ]
    for search_space, expected_values in search_spaces:
        searcher = LocalGridSearcher(search_space=search_space)
        actual_values = []
        while True:
            try:
                cfg = searcher.get_config()
                actual_values.append(cfg)
                searcher.update(cfg, reward=0.1)
            except AssertionError as e:
                assert expected_values == actual_values
                break


def test_local_grid_searcher_numeric_grid_settings():
    search_spaces = [
        [dict(a=space.Int(12, 15)), [{"a": 12}, {"a": 15}]],
        [dict(b=space.Int(12, 15)), [{"b": 12}, {"b": 13}, {"b": 15}]],
        [dict(c=space.Int(12, 15)), [{"c": 12}, {"c": 13}, {"c": 14}, {"c": 15}]],
    ]

    grid_num_sample_settings = {
        "b": 3,
        "c": 4,
    }

    for search_space, expected_values in search_spaces:
        searcher = LocalGridSearcher(
            search_space=search_space,
            grid_numeric_spaces_points_number=2,
            grid_num_sample_settings=grid_num_sample_settings,
        )
        actual_values = []
        while True:
            try:
                cfg = searcher.get_config()
                actual_values.append(cfg)
                searcher.update(cfg, reward=0.1)
            except AssertionError as e:
                assert expected_values == actual_values
                break
