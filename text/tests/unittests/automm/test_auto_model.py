import pytest
from autogluon.text.automm.models import HFAutoModelForTextPrediction
from autogluon.text.automm.models import TimmAutoModelForImagePrediction


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
