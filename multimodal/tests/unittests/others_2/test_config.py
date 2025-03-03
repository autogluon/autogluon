import os

import pytest
from omegaconf import OmegaConf

from autogluon.multimodal.constants import DATA, ENV, MODEL, OPTIM
from autogluon.multimodal.utils import apply_omegaconf_overrides, get_basic_config, get_config, parse_dotlist_conf


def test_basic_config():
    basic_config = get_basic_config()
    assert list(basic_config.keys()).sort() == [MODEL, DATA, OPTIM, ENV].sort()

    basic_config = get_basic_config()
    assert list(basic_config.keys()).sort() == [MODEL, DATA, OPTIM, ENV].sort()


def test_get_config():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_config_path = os.path.join(cur_path, "../../../src/autogluon/multimodal/configs/model/default.yaml")
    model_config = OmegaConf.load(model_config_path)
    data_config_path = os.path.join(cur_path, "../../../src/autogluon/multimodal/configs/data/default.yaml")
    data_config = OmegaConf.load(data_config_path)
    optim_config_path = os.path.join(cur_path, "../../../src/autogluon/multimodal/configs/optim/default.yaml")
    optim_config = OmegaConf.load(optim_config_path)
    environemnt_config_path = os.path.join(cur_path, "../../../src/autogluon/multimodal/configs/env/default.yaml")
    environemnt_config = OmegaConf.load(environemnt_config_path)
    config_gt = OmegaConf.merge(model_config, data_config, optim_config, environemnt_config)

    # test yaml path
    config = {
        MODEL: model_config_path,
        DATA: data_config_path,
        OPTIM: optim_config_path,
        ENV: environemnt_config_path,
    }
    config = get_config(config=config)
    assert config == config_gt

    # test DictConfg
    config = {
        MODEL: model_config,
        DATA: data_config,
        OPTIM: optim_config,
        ENV: environemnt_config,
    }
    config = get_config(config=config)
    assert config == config_gt

    # test dict
    model_config = OmegaConf.to_container(model_config)
    assert isinstance(model_config, dict)

    data_config = OmegaConf.to_container(data_config)
    assert isinstance(data_config, dict)

    optim_config = OmegaConf.to_container(optim_config)
    assert isinstance(optim_config, dict)

    environemnt_config = OmegaConf.to_container(environemnt_config)
    assert isinstance(environemnt_config, dict)

    config = {
        MODEL: model_config,
        DATA: data_config,
        OPTIM: optim_config,
        ENV: environemnt_config,
    }
    config = get_config(config=config)
    assert config == config_gt

    # test default string
    config = {
        MODEL: f"default",
        DATA: "default",
        OPTIM: "default",
        ENV: "default",
    }
    config = get_config(config=config)
    assert config == config_gt


@pytest.mark.parametrize(
    "model_names,",
    [
        ["timm_image"],
        ["hf_text"],
        ["clip"],
        ["timm_image", "hf_text", "clip", "fusion_mlp"],
        ["numerical_mlp", "categorical_mlp", "hf_text", "fusion_mlp"],
        ["numerical_mlp", "categorical_mlp", "timm_image", "fusion_mlp"],
        ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "clip", "fusion_mlp"],
    ],
)
def test_model_config_selection(model_names):
    overrides = {"model.names": model_names}
    config = get_config(overrides=overrides)
    assert sorted(config.model.names) == sorted(model_names)
    names2 = list(config.model.keys())
    names2.remove("names")
    assert sorted(config.model.names) == sorted(names2)


@pytest.mark.parametrize(
    "model_names,",
    [
        ["image"],
        ["text"],
        ["numerical"],
        ["categorical"],
    ],
)
def test_invalid_model_config_selection(model_names):
    overrides = {"model.names": model_names}

    with pytest.raises(ValueError):
        config = get_config(overrides=overrides)


@pytest.mark.parametrize(
    "data,expected",
    [
        ("aaa=a bbb=b ccc=c", {"aaa": "a", "bbb": "b", "ccc": "c"}),
        ("a.a.aa=b b.b.bb=c", {"a.a.aa": "b", "b.b.bb": "c"}),
        ("a.a.aa=1 b.b.bb=100", {"a.a.aa": "1", "b.b.bb": "100"}),
        (["a.a.aa=1", "b.b.bb=100"], {"a.a.aa": "1", "b.b.bb": "100"}),
    ],
)
def test_parse_dotlist_conf(data, expected):
    assert parse_dotlist_conf(data) == expected


def test_apply_omegaconf_overrides():
    conf = OmegaConf.from_dotlist(["a.aa.aaa=[1, 2, 3, 4]", "a.aa.bbb=2", "a.bb.aaa='100'", "a.bb.bbb=4"])
    overrides = "a.aa.aaa=[1, 3, 5] a.aa.bbb=3"
    new_conf = apply_omegaconf_overrides(conf, overrides.split())
    assert new_conf.a.aa.aaa == [1, 3, 5]
    assert new_conf.a.aa.bbb == 3
    new_conf2 = apply_omegaconf_overrides(conf, {"a.aa.aaa": [1, 3, 5, 7], "a.aa.bbb": 4})
    assert new_conf2.a.aa.aaa == [1, 3, 5, 7]
    assert new_conf2.a.aa.bbb == 4

    with pytest.raises(KeyError):
        new_conf3 = apply_omegaconf_overrides(conf, {"a.aa.aaaaaa": [1, 3, 5, 7], "a.aa.bbb": 4})


@pytest.mark.parametrize(
    "overrides",
    [
        {"optim.peft": "None"},
        {"optim.peft": "none"},
        {"optim.peft": "nOne"},
        {"optim.peft": None},
        {"data.label.numerical_preprocessing": None},
        {"data.label.numerical_preprocessing": "none"},
    ],
)
def test_none_str_config(overrides):
    config = get_config(overrides=overrides)
    if "optim.peft" in overrides:
        assert config.optim.peft is None
    if "data.label.numerical_preprocessing" in overrides:
        assert config.data.label.numerical_preprocessing is None
