import pytest
from autogluon.text.automm.presets import get_preset


def test_presets():
    all_types = [
        "text", "image", "clip", "numerical_mlp", "categorical_mlp",
        "fusion_mlp_text_tabular", "fusion_mlp_image_text_tabular",
    ]
    for per_type in all_types:
        config = get_preset(per_type)

    # test cases
    config = get_preset("Text")
    config = get_preset("TEXT")
    config = get_preset("IMAGE")

    # test non-existing types
    non_exist_types = ["hello", "haha"]
    for per_type in non_exist_types:
        with pytest.raises(ValueError):
            config = get_preset(per_type)
