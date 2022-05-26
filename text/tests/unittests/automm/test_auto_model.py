import pytest
from autogluon.text.automm.models import (
    HFAutoModelForTextPrediction,
    TimmAutoModelForImagePrediction,
    NumericalTransformer,
)


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "distilroberta-base",
        "huawei-noah/TinyBERT_General_4L_312D",
        "google/electra-base-discriminator",
        "microsoft/deberta-v3-base",
        "bert-base-uncased",
        "xlm-roberta-base",
        "microsoft/deberta-base",
        "roberta-base",
        "distilbert-base-uncased",
        "bert-base-chinese",
        "gpt2",
    ]
)
def test_hf_automodel_init(checkpoint_name):
    model = HFAutoModelForTextPrediction(
        prefix='model',
        checkpoint_name=checkpoint_name,
        num_classes=5
    )
    # model.get_layer_ids()


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "swin_base_patch4_window7_224",
        "vit_small_patch16_384",
        "resnet18",
        "legacy_seresnet18",
    ]
)
def test_timm_automodel_init(checkpoint_name):
    model = TimmAutoModelForImagePrediction(
        prefix='model',
        checkpoint_name=checkpoint_name,
        num_classes=5
    )
    # model.get_layer_ids()


@pytest.mark.parametrize(
    "embedding_arch",
    [
        ['positional'],
        ['positional','linear'],
        ['linear'],
        ['linear','relu','linear'],
        ['linear','layernorm','relu'],
        ['autodis'],
        ['autodis','linear'],
    ]
)
def test_numerical_transformer_init(embedding_arch):
    import torch
    from autogluon.text.automm.constants import LOGITS, FEATURES
    
    in_features = 10
    d_token = 192
    num_classes = 5

    model = NumericalTransformer(
        prefix='model',
        num_classes=num_classes,
        in_features=in_features,
        d_token=d_token,
        embedding_arch=embedding_arch,
    )

    y = model.forward(
        {
            model.numerical_key: torch.ones(1,in_features), # synthetic data
        }
    )[model.prefix]

    assert y[LOGITS].shape == (1,num_classes) # check the output shape
    assert y[FEATURES].shape == (1,in_features,d_token) # check the output shape
