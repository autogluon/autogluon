import os
import pytest
from omegaconf import OmegaConf
from autogluon.text.automm.utils import get_config
from autogluon.text.automm.constants import (
    MODEL,
    DATA,
    OPTIMIZATION,
    ENVIRONMENT,
)


def test_config():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_config_path = os.path.join(
        cur_path, "../../../src/autogluon/text/automm/configs/model/fusion_mlp_image_text_tabular.yaml"
    )
    model_config = OmegaConf.load(model_config_path)
    data_config_path = os.path.join(
        cur_path, "../../../src/autogluon/text/automm/configs/data/default.yaml"
    )
    data_config = OmegaConf.load(data_config_path)
    optimization_config_path = os.path.join(
        cur_path, "../../../src/autogluon/text/automm/configs/optimization/adamw.yaml"
    )
    optimization_config = OmegaConf.load(optimization_config_path)
    environemnt_config_path = os.path.join(
        cur_path, "../../../src/autogluon/text/automm/configs/environment/default.yaml"
    )
    environemnt_config = OmegaConf.load(environemnt_config_path)
    config_gt = OmegaConf.merge(model_config, data_config, optimization_config, environemnt_config)

    # test yaml path
    config = {
        MODEL: model_config_path,
        DATA: data_config_path,
        OPTIMIZATION: optimization_config_path,
        ENVIRONMENT: environemnt_config_path,
    }
    config = get_config(config)
    assert config == config_gt

    # test DictConfg
    config = {
        MODEL: model_config,
        DATA: data_config,
        OPTIMIZATION: optimization_config,
        ENVIRONMENT: environemnt_config,
    }
    config = get_config(config)
    assert config == config_gt

    # test dict
    model_config = OmegaConf.to_container(model_config)
    assert isinstance(model_config, dict)

    data_config = OmegaConf.to_container(data_config)
    assert isinstance(data_config, dict)

    optimization_config = OmegaConf.to_container(optimization_config)
    assert isinstance(optimization_config, dict)

    environemnt_config = OmegaConf.to_container(environemnt_config)
    assert isinstance(environemnt_config, dict)

    config = {
        MODEL: model_config,
        DATA: data_config,
        OPTIMIZATION: optimization_config,
        ENVIRONMENT: environemnt_config,
    }
    config = get_config(config)
    assert config == config_gt

    # test default string
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    config = get_config(config)
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
    ]
)
def test_model_selection(model_names):
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    overrides = {"model.names": model_names}
    config = get_config(
        config=config,
        overrides=overrides,
    )
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
    ]
)
def test_invalid_model_selection(model_names):
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    overrides = {"model.names": model_names}

    with pytest.raises(ValueError):
        config = get_config(
            config=config,
            overrides=overrides,
        )
