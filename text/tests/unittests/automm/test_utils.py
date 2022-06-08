import pytest
from omegaconf import OmegaConf
from ray import tune
from sklearn.preprocessing import LabelEncoder
from autogluon.text.automm.utils import (
    apply_omegaconf_overrides,
    filter_search_space,
    parse_dotlist_conf,
    get_config,
    try_to_infer_pos_label,
)
from autogluon.text.automm.constants import (
    MODEL,
    DATA,
    OPTIMIZATION,
    ENVIRONMENT,
    BINARY,
    MULTICLASS,
)


@pytest.mark.parametrize('hyperparameters, keys_to_filter, expected',
                         [
                            ({'model.abc':tune.choice(['a','b'])}, ['model'], {}),
                            ({'model.abc':tune.choice(['a','b'])}, ['data'], {'model.abc':tune.choice(['a','b'])}),
                            ({'model.abc':'def'}, ['model'], {'model.abc': 'def'}),
                            ({
                                'data.abc.def':tune.choice(['a','b']),
                                'model.abc':'def',
                                'environment.abc.def':tune.choice(['a','b'])
                            },
                            ['data'],
                            {
                                'model.abc': 'def',
                                'environment.abc.def':tune.choice(['a','b'])
                            }),
                         ])
def test_filter_search_space(hyperparameters, keys_to_filter, expected):
    # We test keys here because the object might be copied and hence direct comparison will fail
    assert filter_search_space(hyperparameters, keys_to_filter).keys() == expected.keys()
    
    
@pytest.mark.parametrize('hyperparameters, keys_to_filter',
                         [
                             ({'model.abc':tune.choice(['a','b'])}, ['abc'])
                         ]
                        )
def test_invalid_filter_search_space(hyperparameters, keys_to_filter):
    with pytest.raises(Exception) as e_info:
        filter_search_space(hyperparameters, keys_to_filter)


@pytest.mark.parametrize('data,expected',
                         [
                             ('aaa=a bbb=b ccc=c', {'aaa': 'a', 'bbb': 'b', 'ccc': 'c'}),
                             ('a.a.aa=b b.b.bb=c', {'a.a.aa': 'b', 'b.b.bb': 'c'}),
                             ('a.a.aa=1 b.b.bb=100', {'a.a.aa': '1', 'b.b.bb': '100'}),
                             (['a.a.aa=1', 'b.b.bb=100'], {'a.a.aa': '1', 'b.b.bb': '100'})
                         ])
def test_parse_dotlist_conf(data, expected):
    assert parse_dotlist_conf(data) == expected


def test_apply_omegaconf_overrides():
    conf = OmegaConf.from_dotlist(["a.aa.aaa=[1, 2, 3, 4]",
                                   "a.aa.bbb=2",
                                   "a.bb.aaa='100'",
                                   "a.bb.bbb=4"])
    overrides = 'a.aa.aaa=[1, 3, 5] a.aa.bbb=3'
    new_conf = apply_omegaconf_overrides(conf, overrides.split())
    assert new_conf.a.aa.aaa == [1, 3, 5]
    assert new_conf.a.aa.bbb == 3
    new_conf2 = apply_omegaconf_overrides(conf, {'a.aa.aaa': [1, 3, 5, 7], 'a.aa.bbb': 4})
    assert new_conf2.a.aa.aaa == [1, 3, 5, 7]
    assert new_conf2.a.aa.bbb == 4

    with pytest.raises(KeyError):
        new_conf3 = apply_omegaconf_overrides(conf, {'a.aa.aaaaaa': [1, 3, 5, 7], 'a.aa.bbb': 4})


@pytest.mark.parametrize(
    "labels,pos_label,problem_type,true_pos_label",
    [
        ([1, 2, 2], 2, BINARY, 1),
        ([1, 2, 2], 1, BINARY, 0),
        ([1, 0, 1, 0, 0], 0, BINARY, 0),
        ([1, 0, 1, 0, 0], None, BINARY, 1),
        (["a", "e", "e", "a"], "a", BINARY, 0),
        (["b", "d", "b", "d"], "d", BINARY, 1),
        (["a", "d", "e", "b"], "d", MULTICLASS, None),
        ([3, 2, 1, 0], 2, MULTICLASS, None),
    ]
)
def test_inferring_pos_label(labels, pos_label, problem_type, true_pos_label):
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    overrides = {}
    if pos_label is not None:
        overrides.update(
            {
                "data.pos_label": pos_label,
            }
        )
    config = get_config(
        config=config,
        overrides=overrides,
    )
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    pos_label = try_to_infer_pos_label(
        data_config=config.data,
        label_encoder=label_encoder,
        problem_type=problem_type,
    )
    assert pos_label == true_pos_label
