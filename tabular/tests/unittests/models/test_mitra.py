from unittest import mock

import pytest

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.mitra.mitra_model import MitraModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"fine_tune_steps": 2}


@pytest.mark.gpu
def test_mitra():
    if ResourceManager.get_gpu_count_torch() == 0:
        # Skip test if no GPU available
        pytest.skip("Skip, no GPU available.")

    model_cls = MitraModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
        # Mitra returns different predictions when predicting on an individual sample
        verify_single_prediction_equivalent_to_multi=False,
    )


@pytest.mark.parametrize(
    "problem_type, hyperparameters, expected",
    [
        # No alias given -> None, so the model class' own default applies.
        ("binary", {}, None),
        ("regression", {}, None),
        # The most specific alias wins: problem-specific > general > generic.
        ("binary", {"hf_cls_model": "org/cls", "hf_general_model": "org/gen"}, "org/cls"),
        ("binary", {"hf_cls_model": "org/cls", "hf_model": "org/generic"}, "org/cls"),
        ("binary", {"hf_general_model": "org/gen", "hf_model": "org/generic"}, "org/gen"),
        ("binary", {"hf_model": "org/generic"}, "org/generic"),
        ("regression", {"hf_reg_model": "org/reg", "hf_cls_model": "org/cls"}, "org/reg"),
        ("regression", {"hf_general_model": "org/gen"}, "org/gen"),
        # An alias for the other problem type is ignored.
        ("binary", {"hf_reg_model": "org/reg"}, None),
        ("regression", {"hf_cls_model": "org/cls"}, None),
    ],
)
def test_mitra_resolve_hf_model(problem_type, hyperparameters, expected):
    assert MitraModel._resolve_hf_model(problem_type=problem_type, hyp=dict(hyperparameters)) == expected


def test_mitra_resolve_hf_model_pops_every_alias():
    """Every alias must be popped, else an unused one leaks into the model constructor."""
    hyp = {
        "hf_cls_model": "org/cls",
        "hf_reg_model": "org/reg",
        "hf_general_model": "org/gen",
        "hf_model": "org/generic",
        "n_estimators": 1,
    }
    assert MitraModel._resolve_hf_model(problem_type="binary", hyp=hyp) == "org/cls"
    assert hyp == {"n_estimators": 1}


def test_mitra_resolve_hf_model_rejects_unsupported_problem_type():
    with pytest.raises(AssertionError, match="Unsupported problem_type"):
        MitraModel._resolve_hf_model(problem_type="quantile", hyp={})


def test_mitra_loads_local_checkpoint_dir(tmp_path):
    """A local directory written by `save_pretrained` is loaded from disk instead of HuggingFace."""
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    checkpoint_dir = tmp_path / "checkpoint"
    Tab2D(
        dim=32,
        dim_output=10,
        n_layers=1,
        n_heads=2,
        task="CLASSIFICATION",
        use_pretrained_weights=False,
        path_to_weights="",
        device="cpu",
    ).save_pretrained(str(checkpoint_dir))
    assert (checkpoint_dir / "config.json").is_file()
    assert (checkpoint_dir / "model.safetensors").is_file()

    with mock.patch(
        "autogluon.tabular.models.mitra._internal.models.tab2d.hf_hub_download",
        side_effect=AssertionError("must not hit HuggingFace for a local checkpoint"),
    ):
        loaded = Tab2D.from_pretrained(str(checkpoint_dir), device="cpu")
    assert loaded.dim == 32
    assert loaded.n_layers == 1


def test_mitra_local_checkpoint_dir_missing_file(tmp_path):
    """An incomplete local checkpoint directory reports the missing file instead of a download error."""
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="config.json"):
        Tab2D.from_pretrained(str(tmp_path / "empty"), device="cpu")
