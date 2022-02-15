from omegaconf import OmegaConf
from autogluon.text.automm.utils import get_config
from autogluon.text.automm.constants import (
    MODEL,
    DATA,
    OPTIMIZATION,
    ENVIRONMENT,
)


def test_config():
    model_config = OmegaConf.load("configs/model/fusion_mlp_image_text_tabular.yaml")
    data_config = OmegaConf.load("configs/data/default.yaml")
    optimization_config = OmegaConf.load("configs/optimization/adamw.yaml")
    environemnt_config = OmegaConf.load("configs/environment/default.yaml")
    config_gt = OmegaConf.merge(model_config, data_config, optimization_config, environemnt_config)

    # test yaml path
    config = {
        MODEL: f"configs/model/fusion_mlp_image_text_tabular.yaml",
        DATA: "configs/data/default.yaml",
        OPTIMIZATION: "configs/optimization/adamw.yaml",
        ENVIRONMENT: "configs/environment/default.yaml",
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




