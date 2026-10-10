import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from autogluon.core.models import BaggedEnsembleModel
from autogluon.tabular.models.mitra.checkpoint import CHECKPOINTS, validate_checkpoint_version
from autogluon.tabular.models.mitra.mitra_model import MitraModel


@pytest.fixture
def checkpoint_factory(tmp_path):
    import torch

    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    torch.manual_seed(0)

    def make(task="CLASSIFICATION", dim_output=10, version=None):
        directory = tmp_path / f"checkpoint_{task}_{dim_output}_{version}"
        model = Tab2D(
            dim=16,
            dim_output=dim_output,
            n_layers=1,
            n_heads=2,
            task=task,
            use_pretrained_weights=False,
            path_to_weights="",
            device="cpu",
            mitra_version=version,
        )
        model.save_pretrained(str(directory))
        return str(directory)

    return make


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("version", [None, "1", "2"])
def test_version_selects_estimator_and_checkpoint(problem_type, version):
    model = MitraModel(
        problem_type=problem_type,
        hyperparameters={} if version is None else {"model_version": version},
    )
    expected_version = version or "2"
    estimator = model.get_model_cls()(model_version=expected_version)
    task = "regression" if problem_type == "regression" else "classification"
    assert estimator.model_version == expected_version
    assert estimator.hf_model == CHECKPOINTS[expected_version][task]
    assert estimator.fine_tune_recipe == "2"
    assert model._get_fine_tune_recipe() == "2"


@pytest.mark.parametrize("version", ["1", "2"])
def test_unspecified_recipe_uses_new_method_for_both_weights(version):
    model = MitraModel(
        problem_type="binary",
        hyperparameters={"model_version": version, "fine_tune_recipe": None},
    )
    estimator = model.get_model_cls()(model_version=version, fine_tune_recipe=None)
    assert model._get_fine_tune_recipe() == "2"
    assert estimator.fine_tune_recipe == "2"
    assert estimator.warmup_steps == 10
    assert estimator.weight_decay == 0.3


def test_versions_share_the_finetuning_implementation():
    v1 = MitraModel(problem_type="binary", hyperparameters={"model_version": "1"})
    v2 = MitraModel(problem_type="binary", hyperparameters={"model_version": "2"})
    assert v1.get_model_cls() is v2.get_model_cls()
    legacy = v1.get_model_cls()(model_version="1", fine_tune_recipe="1")
    shared = v2.get_model_cls()(model_version="2", fine_tune_recipe="2")
    assert legacy._get_trainer_cls() is shared._get_trainer_cls()


@pytest.mark.parametrize("recipe", ["auto", "3", False, 2])
def test_invalid_recipe_fails_early(recipe):
    model = MitraModel(problem_type="binary", hyperparameters={"fine_tune_recipe": recipe})
    with pytest.raises(ValueError, match="fine_tune_recipe"):
        model.get_model_cls()


@pytest.mark.parametrize("version", ["auto", "3", 2, None, [], True])
def test_invalid_version_fails_early(version):
    model = MitraModel(problem_type="binary", hyperparameters={"model_version": version})
    with pytest.raises(ValueError, match="model_version"):
        model.get_model_cls()


@pytest.mark.parametrize("version", ["1", "2"])
def test_download_weights_uses_selected_version(version):
    with mock.patch.object(MitraModel, "download_weights") as download:
        MitraModel.download_default_weights(model_version=version)
    assert download.call_args_list == [
        mock.call(repo_id=CHECKPOINTS[version]["classification"]),
        mock.call(repo_id=CHECKPOINTS[version]["regression"]),
    ]


def test_checkpoint_version_is_not_inferred_from_architecture(checkpoint_factory):
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    v1 = checkpoint_factory(version="1")
    v2 = checkpoint_factory(version="2")
    with open(f"{v1}/config.json") as file:
        config1 = json.load(file)
    with open(f"{v2}/config.json") as file:
        config2 = json.load(file)
    assert config1.pop("mitra_version") == "1"
    assert config2.pop("mitra_version") == "2"
    assert config1 == config2
    with pytest.raises(ValueError, match="model_version='1'"):
        Tab2D.from_pretrained(v1, device="cpu", model_version="2")
    with pytest.raises(ValueError, match="model_version='2'"):
        Tab2D.from_pretrained(v2, device="cpu", model_version="1")
    assert Tab2D.from_pretrained(v2, device="cpu", model_version="2").mitra_version == "2"


@pytest.mark.parametrize("version", ["1", "2"])
def test_unlabelled_local_checkpoint_obeys_explicit_selection(checkpoint_factory, version):
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    path = checkpoint_factory()
    loaded = Tab2D.from_pretrained(path, device="cpu", model_version=version)
    assert loaded.mitra_version is None  # Selecting a version does not invent weight provenance.


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_known_checkpoint_mismatch_rejected(task):
    with pytest.raises(ValueError, match="model_version='1'"):
        validate_checkpoint_version(CHECKPOINTS["1"][task], "2")
    with pytest.raises(ValueError, match="model_version='2'"):
        validate_checkpoint_version(CHECKPOINTS["2"][task], "1")


@pytest.mark.parametrize(
    "task,n_classes,n_rows,expected_lr,expected_support,expected_predict_support",
    [
        ("classification", 2, 100, 3e-6, 16384, 16384),
        ("classification", 2, 16384, 3e-6, 16384, 16384),
        ("classification", 2, 16385, 1e-5, 16384, 16384),
        ("classification", 3, 100, 1e-5, 16384, 32768),
        ("regression", 0, 100, 1e-5, 20480, 32768),
    ],
)
def test_v2_recipe(task, n_classes, n_rows, expected_lr, expected_support, expected_predict_support):
    from autogluon.tabular.models.mitra.finetune import MitraFinetuneClassifier, MitraFinetuneRegressor

    estimator = (MitraFinetuneRegressor if task == "regression" else MitraFinetuneClassifier)()
    estimator.y = np.arange(10) if task == "regression" else np.arange(10) % n_classes
    estimator._num_samples_total = n_rows
    cfg, _ = estimator._create_config(task, 10, time_limit=100)
    assert cfg.hyperparams["lr"] == expected_lr
    assert cfg.hyperparams["warmup_steps"] == 10
    assert cfg.hyperparams["weight_decay"] == 0.3
    assert cfg.hyperparams["max_epochs"] == 50
    assert cfg.hyperparams["max_samples_support"] == expected_support
    assert cfg.hyperparams["predict_max_samples_support"] == expected_predict_support
    assert cfg.hyperparams["predict_max_samples_query"] == 16384


def test_v1_recipe_is_unchanged():
    from autogluon.tabular.models.mitra.finetune import MitraFinetuneClassifier
    from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier

    cfg, _ = MitraClassifier()._create_config("classification", 10)
    assert cfg.hyperparams["lr"] == 1e-4
    assert cfg.hyperparams["warmup_steps"] == 1000
    assert cfg.hyperparams["weight_decay"] == 0.1
    assert cfg.hyperparams["max_samples_support"] == 8192
    assert cfg.hyperparams["max_samples_query"] == 1024
    shared = MitraFinetuneClassifier(model_version="1", fine_tune_recipe="1")
    shared.y = np.arange(20) % 2
    shared_cfg, _ = shared._create_config("classification", 10)
    for key in ("lr", "warmup_steps", "weight_decay", "max_samples_support", "max_samples_query"):
        assert shared_cfg.hyperparams[key] == cfg.hyperparams[key]


@pytest.mark.parametrize("task,n_classes", [("CLASSIFICATION", 2), ("CLASSIFICATION", 3), ("REGRESSION", 0)])
def test_legacy_recipe_matches_legacy_estimator(checkpoint_factory, task, n_classes):
    import torch

    from autogluon.tabular.models.mitra.finetune import MitraFinetuneClassifier, MitraFinetuneRegressor
    from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier, MitraRegressor

    path = checkpoint_factory(task=task, dim_output=10 if n_classes else 1, version="1")
    rng = np.random.RandomState(1)
    X = rng.normal(size=(48, 4))
    y = np.arange(48) % n_classes if n_classes else rng.normal(size=48)
    kwargs = {"hf_model": path, "device": "cpu", "fine_tune_steps": 3, "verbose": False}
    legacy = (MitraClassifier if n_classes else MitraRegressor)(**kwargs)
    shared = (MitraFinetuneClassifier if n_classes else MitraFinetuneRegressor)(
        **kwargs, model_version="1", fine_tune_recipe="1"
    )
    predictions = []
    for estimator in (legacy, shared):
        torch.manual_seed(0)
        np.random.seed(0)
        estimator.fit(X[:32], y[:32], X_val=X[32:], y_val=y[32:])
        predict = estimator.predict_proba if n_classes else estimator.predict
        predictions.append(predict(X[32:]))
    np.testing.assert_allclose(predictions[0], predictions[1], atol=1e-6)
    for name, value in legacy.trainers[0].model.state_dict().items():
        torch.testing.assert_close(value, shared.trainers[0].model.state_dict()[name])


def test_explicit_recipe_overrides_and_time_budget():
    from autogluon.tabular.models.mitra.finetune import MitraFinetuneClassifier

    estimator = MitraFinetuneClassifier(lr=2e-5, weight_decay=0.2, warmup_steps=3, fine_tune_time_limit=250)
    estimator.y = np.arange(20) % 2
    cfg, _ = estimator._create_config("classification", 10, time_limit=100)
    assert cfg.hyperparams["lr"] == 2e-5
    assert cfg.hyperparams["weight_decay"] == 0.2
    assert cfg.hyperparams["warmup_steps"] == 3
    assert cfg.hyperparams["budget"] == 100
    cfg, _ = estimator._create_config("classification", 10, time_limit=1000)
    assert cfg.hyperparams["budget"] == 250


def test_regression_decodes_mean_instead_of_mode():
    import torch

    from autogluon.tabular.models.mitra._internal.core.trainer_finetune_v2 import TrainerFinetuneV2

    trainer = TrainerFinetuneV2.__new__(TrainerFinetuneV2)
    trainer.bins = torch.tensor([-0.5, 0.5, 1.5])
    trainer.bin_width = trainer.bins[1] - trainer.bins[0]
    logits = torch.log(torch.tensor([[0.25, 0.75], [0.6, 0.4]]))
    torch.testing.assert_close(trainer._decode_regression(logits), torch.tensor([0.75, 0.4]))


@pytest.mark.gpu
def test_regression_prediction_matches_released_cpu_mean():
    import torch

    from autogluon.tabular.models.mitra._internal.core.trainer_finetune_v2 import TrainerFinetuneV2

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to check prediction from GPU logits.")
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn((1, 67, 1000), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    logits[0, 0, 0] = float("nan")
    logits[0, 1, 1] = float("inf")
    logits[0, 2, 2] = -float("inf")

    trainer = TrainerFinetuneV2.__new__(TrainerFinetuneV2)
    trainer.device = "cuda"
    trainer.cfg = SimpleNamespace(
        task="regression",
        hyperparams={
            "regression_loss": "cross_entropy",
            "precision": "bfloat16",
            "fast_predict": True,
            "balance_binary_support": False,
        },
    )
    trainer.bins = torch.linspace(-0.5, 1.5, 1001, device="cuda")
    trainer.bin_width = trainer.bins[1] - trainer.bins[0]
    trainer.rng = np.random.RandomState(0)
    trainer.model = mock.Mock()
    trainer.preprocessor = SimpleNamespace(inverse_transform_y=lambda values: values)
    trainer.make_loader = lambda dataset, training: [{}]
    trainer._forward = lambda batch: logits

    actual = trainer._predict_transformed(np.zeros((2, 1)), np.zeros(2), np.zeros((67, 1)), 2, 67, prediction=True)
    # The released API transfers logits before its float32 softmax and mean.
    cpu_logits = torch.nan_to_num(logits[0].float().cpu(), nan=0.0, posinf=1e4, neginf=-1e4)
    cpu_centers = (trainer.bins[:-1] + trainer.bin_width / 2).cpu()
    expected = (cpu_logits.softmax(dim=-1) * cpu_centers).sum(dim=-1).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_device_checkpoint_does_not_alias_training_weights():
    import torch

    from autogluon.tabular.models.mitra._internal.core.trainer_finetune_v2 import DeviceCheckpoint

    model = torch.nn.Linear(2, 1)
    checkpoint = DeviceCheckpoint()
    checkpoint.reset(model)
    checkpoint(model, 1.0)
    expected = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(10)
    checkpoint(model, 2.0)
    checkpoint.set_to_best(model)
    torch.testing.assert_close(model.weight, expected)


@pytest.mark.parametrize(
    "counts,expected",
    [
        ([2, 8], [0, 1, 8, 4, 3, 9]),
        ([8, 2], [8, 9, 6, 2, 1, 7]),
        ([5, 5], [2, 0, 1, 5, 7, 6]),
    ],
)
def test_balanced_prediction_support_matches_released_recipe(counts, expected):
    from autogluon.tabular.models.mitra._internal.core.trainer_finetune_v2 import PredictionDataset

    rng = np.random.RandomState(0)
    dataset = PredictionDataset(
        cfg=None,
        x_support=np.arange(10)[:, None],
        y_support=np.repeat([0, 1], counts),
        x_query=np.zeros((2, 1)),
        y_query=None,
        max_samples_support=6,
        max_samples_query=1,
        rng=rng,
        balance_binary=True,
    )
    # Frozen draws from the released recipe, including its ordering and RNG
    # consumption when the minority class is included in full.
    np.testing.assert_array_equal(dataset[0]["x_support"].numpy().ravel(), expected)
    np.testing.assert_array_equal(rng.randint(0, 1_000_000, size=3), [610581, 448242, 374564])


@pytest.mark.parametrize("version", ["1", "2"])
@pytest.mark.parametrize("recipe", ["1", "2"])
@pytest.mark.parametrize("problem_type,n_classes", [("binary", 2), ("multiclass", 3), ("regression", 0)])
def test_fit_predict_save_load(checkpoint_factory, tmp_path, version, recipe, problem_type, n_classes):
    task = "REGRESSION" if problem_type == "regression" else "CLASSIFICATION"
    output = (7 if version == "2" else 1) if problem_type == "regression" else 10
    path = checkpoint_factory(task=task, dim_output=output, version=version)
    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.normal(size=(48, 4)), columns=list("abcd"))
    y = pd.Series(rng.normal(size=48) if n_classes == 0 else np.arange(48) % n_classes)
    model = MitraModel(
        path=str(tmp_path / "model"),
        problem_type=problem_type,
        hyperparameters={"model_version": version, "fine_tune_recipe": recipe, "hf_model": path, "fine_tune_steps": 2},
    )
    model.fit(X=X.iloc[:32], y=y.iloc[:32], X_val=X.iloc[32:], y_val=y.iloc[32:], num_cpus=1, num_gpus=0)
    # Save before predicting so the restored model starts with the same support RNG.
    model.save()
    prediction = model.predict_proba(X.iloc[32:])
    assert np.isfinite(prediction).all()
    assert prediction.shape == ((16, n_classes) if problem_type == "multiclass" else (16,))
    if n_classes > 0:
        assert ((prediction >= 0) & (prediction <= 1)).all()
    if version == "2" and problem_type == "regression":
        assert model.model.trainers[0].cfg.hyperparams["dim_output"] == 7
        assert model.model.trainers[0].cfg.hyperparams["regression_loss"] == "cross_entropy"
    if version == "1" and problem_type == "regression":
        assert model.model.trainers[0].cfg.hyperparams["regression_loss"] == "mse"
    assert model.model.fine_tune_recipe == recipe
    loaded = MitraModel.load(model.path)
    np.testing.assert_allclose(loaded.predict_proba(X.iloc[32:]), prediction, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "problem_type,version,output", [("binary", "1", 10), ("regression", "1", 1), ("regression", "2", 7)]
)
def test_gpu_model_loads_and_predicts_without_cuda(checkpoint_factory, tmp_path, problem_type, version, output):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to create a GPU-trained artifact.")
    task = "REGRESSION" if problem_type == "regression" else "CLASSIFICATION"
    checkpoint = checkpoint_factory(task=task, dim_output=output, version=version)
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=(32, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.normal(size=32) if problem_type == "regression" else np.arange(32) % 2)
    model = MitraModel(
        path=str(tmp_path / "model"),
        problem_type=problem_type,
        hyperparameters={
            "hf_model": checkpoint,
            "model_version": version,
            "fine_tune_recipe": "2",
            "fine_tune_steps": 1,
        },
    )
    model.fit(X=X.iloc[:24], y=y.iloc[:24], X_val=X.iloc[24:], y_val=y.iloc[24:], num_cpus=1, num_gpus=1)
    model.save()
    trainer = model.model.trainers[0]
    assert next(trainer.model.parameters()).device.type == "cuda"
    if hasattr(trainer, "bins"):
        assert trainer.bins.device.type == trainer.bin_width.device.type == "cuda"

    query_path = tmp_path / "query.pkl"
    output_path = tmp_path / "prediction.npy"
    X.iloc[24:].to_pickle(query_path)
    # This also checks moving an already fitted GPU network to CPU in-process.
    model.set_device("cpu")
    expected = model.predict_proba(X.iloc[24:])
    script = """
import sys
import numpy as np
import pandas as pd
import torch
from autogluon.tabular.models.mitra.mitra_model import MitraModel
assert not torch.cuda.is_available()
model = MitraModel.load(sys.argv[1])
assert str(model.get_device()) == "cpu"
np.save(sys.argv[3], model.predict_proba(pd.read_pickle(sys.argv[2])))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, model.path, str(query_path), str(output_path)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_allclose(np.load(output_path), expected, atol=1e-5)


def test_failed_save_restores_the_fitted_model(checkpoint_factory, tmp_path):
    from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

    checkpoint = checkpoint_factory(task="REGRESSION", dim_output=7, version="2")
    X = pd.DataFrame(np.random.RandomState(0).normal(size=(24, 2)))
    y = pd.Series(np.arange(24, dtype=float))
    model = MitraModel(
        path=str(tmp_path / "failed_save"),
        problem_type="regression",
        hyperparameters={"hf_model": checkpoint, "fine_tune": False},
    )
    model.fit(X=X.iloc[:16], y=y.iloc[:16], X_val=X.iloc[16:], y_val=y.iloc[16:], num_cpus=1, num_gpus=0)
    trainer = model.model.trainers[0]
    network, bins, width = trainer.model, trainer.bins, trainer.bin_width
    with mock.patch.object(AbstractTorchModel, "save", side_effect=OSError("test disk failure")):
        with pytest.raises(OSError, match="test disk failure"):
            model.save()
    assert trainer.model is network
    assert trainer.bins is bins
    assert trainer.bin_width is width
    assert np.isfinite(model.predict_proba(X.iloc[16:])).all()


@pytest.mark.parametrize("recipe_field", ["absent", "implicit"])
def test_loading_legacy_predictor_does_not_change_recipe(checkpoint_factory, tmp_path, recipe_field):
    path = checkpoint_factory(version="1")
    X = pd.DataFrame(np.random.RandomState(0).normal(size=(24, 2)))
    y = pd.Series(np.arange(24) % 2)
    model = MitraModel(
        path=str(tmp_path / "legacy"),
        problem_type="binary",
        hyperparameters={"model_version": "1", "fine_tune_recipe": "1", "hf_model": path, "fine_tune": False},
    )
    model.fit(X=X.iloc[:16], y=y.iloc[:16], X_val=X.iloc[16:], y_val=y.iloc[16:], num_cpus=1, num_gpus=0)
    if recipe_field == "absent":
        # These fields did not exist on predictors saved before version selection.
        del model.params["model_version"]
        del model._user_params["model_version"]
        del model.params["fine_tune_recipe"]
        del model._user_params["fine_tune_recipe"]
    else:
        # An earlier native predictor resolved None to the checkpoint's recipe.
        model.params["fine_tune_recipe"] = None
        model._user_params["fine_tune_recipe"] = None
    del model._feature_reducer
    model.save()
    prediction = model.predict_proba(X.iloc[16:])
    loaded = MitraModel.load(model.path)
    assert loaded.model.model_version == "1"
    assert loaded.model.fine_tune_recipe == "1"
    assert loaded._get_fine_tune_recipe() == "1"
    assert loaded.convert_to_refit_full_template()._get_model_version() == "1"
    assert loaded.convert_to_refit_full_template()._get_fine_tune_recipe() == "1"
    np.testing.assert_allclose(loaded.predict_proba(X.iloc[16:]), prediction, atol=1e-5)


@pytest.mark.parametrize("heldout_in_support", [True, False])
def test_bagging_extends_support_only_after_oof(checkpoint_factory, tmp_path, heldout_in_support):
    from autogluon.tabular.models.mitra.finetune import MitraFinetuneClassifier

    path = checkpoint_factory(version="2")
    X = pd.DataFrame(np.random.RandomState(3).normal(size=(32, 3)))
    y = pd.Series(np.arange(32) % 2)
    bag = BaggedEnsembleModel(
        model_base=MitraModel(
            problem_type="binary",
            hyperparameters={"hf_model": path, "fine_tune": False, "heldout_in_support": heldout_in_support},
        ),
        path=str(tmp_path / "bag"),
        hyperparameters={"fold_fitting_strategy": "sequential_local"},
    )
    supports_at_prediction = []
    original = MitraFinetuneClassifier.predict_proba

    def predict(estimator, query):
        supports_at_prediction.append(len(estimator.X))
        return original(estimator, query)

    with mock.patch.object(MitraFinetuneClassifier, "predict_proba", predict):
        bag.fit(X=X, y=y, k_fold=2, num_cpus=1, num_gpus=0)
    assert supports_at_prediction
    assert set(supports_at_prediction) == {16}
    assert np.isfinite(bag.predict_proba_oof()).all()
    assert len(bag.models) == 2
    for child in bag.models:
        child = bag.load_child(child)
        assert len(child.model.X) == (32 if heldout_in_support else 16)
        assert child.model._num_samples_total == 32
    assert np.isfinite(bag.predict_proba(X.iloc[:3])).all()


@pytest.mark.parametrize("heldout_in_support", [True, False])
def test_ray_oof_cleanup_finalizes_mitra_support_once(checkpoint_factory, tmp_path, heldout_in_support):
    from autogluon.core.models.ensemble.fold_fitting_strategy import _ray_predict_oof

    X = pd.DataFrame(np.random.RandomState(3).normal(size=(32, 3)))
    y = pd.Series(np.arange(32) % 2)
    model = MitraModel(
        path=str(tmp_path / "fold"),
        problem_type="binary",
        hyperparameters={
            "hf_model": checkpoint_factory(version="2"),
            "fine_tune": False,
            "heldout_in_support": heldout_in_support,
        },
    )
    # Match the existing bagging protocol: validate the template against the
    # full dataset, then fit the already-validated child on its training fold.
    model.validate_fit_args(X=X)
    model._fit_constraints_validated_upstream = True
    model.fit(X=X.iloc[:16], y=y.iloc[:16], X_val=X.iloc[16:], y_val=y.iloc[16:], num_cpus=1, num_gpus=0)
    assert model.model._num_samples_total == 32
    assert len(model.model.X) == 16

    # Cleanup before the fold has a validation score must not reveal labels.
    model.reduce_memory_size()
    assert len(model.model.X) == 16
    result, probabilities = _ray_predict_oof(model, X.iloc[16:], y.iloc[16:], num_cpus=1)
    assert result is model
    assert np.isfinite(probabilities).all()
    assert model.val_score == pytest.approx(model.score_with_y_pred_proba(y.iloc[16:], probabilities))
    assert len(model.model.X) == (32 if heldout_in_support else 16)
    assert model._pending_fold_support is None

    # Saving or reducing an already finalized model cannot append rows twice.
    model.reduce_memory_size()
    assert len(model.model.X) == (32 if heldout_in_support else 16)
    restored = MitraModel.load(model.save())
    assert len(restored.model.X) == (32 if heldout_in_support else 16)
    assert restored._pending_fold_support is None


def test_standalone_cleanup_does_not_add_external_validation(checkpoint_factory, tmp_path):
    X = pd.DataFrame(np.random.RandomState(3).normal(size=(32, 3)))
    y = pd.Series(np.arange(32) % 2)
    model = MitraModel(
        path=str(tmp_path / "standalone"),
        problem_type="binary",
        hyperparameters={"hf_model": checkpoint_factory(version="2"), "fine_tune": False},
    )
    model.fit(X=X.iloc[:16], y=y.iloc[:16], X_val=X.iloc[16:], y_val=y.iloc[16:], num_cpus=1, num_gpus=0)
    probabilities = model.predict_proba(X.iloc[16:])
    model.val_score = model.score_with_y_pred_proba(y.iloc[16:], probabilities)
    model.reduce_memory_size()
    assert model.model._num_samples_total == 16
    assert len(model.model.X) == 16
    assert model._pending_fold_support is None


def test_many_class_fit_is_reusable_and_serializable(checkpoint_factory, tmp_path):
    path = checkpoint_factory(version="2")
    rng = np.random.RandomState(4)
    X = pd.DataFrame(rng.normal(size=(66, 3)))
    y = pd.Series(np.tile(np.arange(11), 6))
    model = MitraModel(
        path=str(tmp_path / "hierarchy"),
        problem_type="multiclass",
        hyperparameters={"hf_model": path, "fine_tune_steps": 1},
    )
    model.fit(X=X.iloc[:44], y=y.iloc[:44], X_val=X.iloc[44:], y_val=y.iloc[44:], num_cpus=1, num_gpus=0)
    model.save()
    prediction = model.predict_proba(X.iloc[44:])
    assert prediction.shape == (22, 11)
    np.testing.assert_allclose(prediction.sum(axis=1), 1, atol=1e-6)
    assert len(model.model.hierarchy_.nodes) == 3
    loaded = MitraModel.load(model.path)
    np.testing.assert_allclose(loaded.predict_proba(X.iloc[44:]), prediction, atol=1e-5)


def test_feature_reduction_fits_only_training_data():
    from autogluon.tabular.models.mitra._internal.data.feature_reduction import FeatureReducer

    rng = np.random.RandomState(5)
    X = pd.DataFrame(rng.normal(size=(40, 300)))
    y = (X[0] > 0).astype(int)
    selector = FeatureReducer("binary").fit(X, y)
    selected = selector.transform(X)
    assert selected.shape == (40, 256)
    assert 0 in selected.columns
    selected_columns = selector.columns.copy()
    selector.transform(X + 10000)
    assert selector.columns == selected_columns
    regression = FeatureReducer("regression").fit(X, y)
    assert regression.transform(X.iloc[:5] + 10000).shape == (5, 40)
    assert np.isfinite(regression.transform(X)).all().all()


def test_binary_descriptor_and_categorical_tables_are_not_reduced():
    from autogluon.tabular.models.mitra._internal.data.feature_reduction import FeatureReducer

    X = pd.DataFrame(np.random.RandomState(6).randint(0, 2, size=(20, 300)))
    y = np.arange(20) % 2
    assert FeatureReducer("binary").fit(X, y).transform(X).shape[1] == 300
    assert FeatureReducer("regression", has_categoricals=True).fit(X, y).transform(X).shape[1] == 300


def test_bagged_regression_matches_released_projection(checkpoint_factory, tmp_path):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from autogluon.tabular.models.mitra.finetune import MitraFinetuneRegressor

    rng = np.random.RandomState(17)
    X = pd.DataFrame(rng.normal(size=(40, 6)), columns=list("abcdef"))
    # A column can be constant within a fold and still be needed by the
    # projection fitted on the outer table.
    X["rare"] = 0.0
    X.loc[0, "rare"] = 1.0
    y = pd.Series(rng.normal(size=len(X)))
    reference = make_pipeline(
        SimpleImputer(strategy="mean"), StandardScaler(), PCA(n_components=3, random_state=0)
    )
    expected = pd.DataFrame(reference.fit_transform(X.to_numpy(dtype=float)), index=X.index)
    bag = BaggedEnsembleModel(
        model_base=MitraModel(
            problem_type="regression",
            hyperparameters={
                "hf_model": checkpoint_factory(task="REGRESSION", dim_output=7, version="2"),
                "fine_tune": False,
                "max_features": 3,
            },
        ),
        path=str(tmp_path / "projection_bag"),
        hyperparameters={"fold_fitting_strategy": "sequential_local"},
    )
    original_fit = MitraFinetuneRegressor.fit
    folds_seen = []

    def fit(estimator, X, y, X_val=None, y_val=None, **kwargs):
        np.testing.assert_array_equal(X.to_numpy(), expected.loc[X.index].to_numpy())
        np.testing.assert_array_equal(X_val.to_numpy(), expected.loc[X_val.index].to_numpy())
        folds_seen.append(X.index.tolist())
        return original_fit(estimator, X, y, X_val=X_val, y_val=y_val, **kwargs)

    with mock.patch.object(MitraFinetuneRegressor, "fit", fit):
        bag.fit(X=X, y=y, k_fold=2, num_cpus=1, num_gpus=0)
    assert len(folds_seen) == 2
    assert np.isfinite(bag.predict_proba_oof()).all()
    query = X.iloc[:4] + 100
    # Reusing training indices must not reuse cached training values at
    # inference, including after loading a saved child.
    for child_name in bag.models:
        child = bag.load_child(child_name)
        assert child._feature_reducer._training_projection is None
        assert not child._use_training_projection
        np.testing.assert_array_equal(child.preprocess(query).to_numpy(), reference.transform(query.to_numpy()))
        child.save()
        restored = MitraModel.load(child.path)
        np.testing.assert_array_equal(restored.preprocess(query).to_numpy(), reference.transform(query.to_numpy()))


def test_projection_cache_checks_row_values_and_preserves_fit_transform():
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from autogluon.tabular.models.mitra._internal.data.feature_reduction import FeatureReducer

    X = pd.DataFrame(np.random.RandomState(7).normal(size=(80, 24)))
    X.index = np.arange(len(X)) // 2
    reference = make_pipeline(
        SimpleImputer(strategy="mean"), StandardScaler(), PCA(n_components=5, random_state=0)
    )
    expected = reference.fit_transform(X.to_numpy())
    reducer = FeatureReducer("regression", max_features=5).fit(X, None, cache_training=True)
    rows = [9, 3, 44, 1]
    np.testing.assert_array_equal(reducer.transform_training(X.iloc[rows]), expected[rows])
    with pytest.raises(ValueError, match="outside its fitted training table"):
        reducer.transform_training(X.iloc[rows] + 1)
    reducer.clear_training_cache()
    np.testing.assert_array_equal(reducer.transform(X.iloc[rows] + 1), reference.transform((X.iloc[rows] + 1).to_numpy()))


def test_prediction_oom_shrinks_queries_before_support():
    from types import SimpleNamespace

    import torch

    from autogluon.tabular.models.mitra._internal.core.trainer_finetune_v2 import TrainerFinetuneV2

    trainer = TrainerFinetuneV2.__new__(TrainerFinetuneV2)
    trainer.cfg = SimpleNamespace(
        hyperparams={
            "predict_max_samples_support": 200,
            "predict_min_samples_support": 50,
            "predict_max_samples_query": 4096,
            "predict_min_samples_query": 256,
        }
    )
    trainer.preprocessor = SimpleNamespace(transform_X=lambda x: x, transform_y=lambda y: y)
    trainer.rng = np.random.RandomState(0)
    attempts = []
    draws = []

    def predict(x_support, y_support, x_query, support_cap, query_chunk, **kwargs):
        attempts.append((support_cap, query_chunk))
        draws.append(trainer.rng.rand())
        if support_cap > 50:
            raise torch.cuda.OutOfMemoryError("test allocation")
        return np.zeros(len(x_query))

    trainer._predict_transformed = predict
    with mock.patch("torch.cuda.empty_cache"):
        prediction = trainer.predict(np.zeros((100, 2)), np.zeros(100), np.zeros((3, 2)))
    assert prediction.shape == (3,)
    assert attempts == [(200, 4096), (200, 2048), (200, 1024), (200, 512), (200, 256), (100, 256), (50, 256)]
    assert len(set(draws)) == 1


@pytest.mark.parametrize("fails_again", [False, True])
def test_final_prediction_oom_retry_releases_traceback(fails_again):
    import sys
    import weakref
    from types import SimpleNamespace

    import torch

    from autogluon.tabular.models.mitra._internal.core.trainer_finetune_v2 import TrainerFinetuneV2

    trainer = TrainerFinetuneV2.__new__(TrainerFinetuneV2)
    trainer.cfg = SimpleNamespace(
        hyperparams={
            "predict_max_samples_support": 8,
            "predict_min_samples_support": 8,
            "predict_max_samples_query": 2,
            "predict_min_samples_query": 2,
        }
    )
    trainer.preprocessor = SimpleNamespace(transform_X=lambda x: x, transform_y=lambda y: y)
    trainer.rng = np.random.RandomState(0)
    attempts, draws, buffers = [], [], []

    def predict(x_support, y_support, x_query, support_cap, query_chunk, **kwargs):
        attempts.append((support_cap, query_chunk))
        draws.append(trainer.rng.rand())
        if len(attempts) == 1:
            # An exception traceback retains the failed forward's local tensors.
            buffer = torch.empty(1)
            buffers.append(weakref.ref(buffer))
            raise torch.cuda.OutOfMemoryError("first allocation")
        assert buffers[0]() is None
        if fails_again:
            raise torch.cuda.OutOfMemoryError("second allocation")
        return np.zeros(len(x_query))

    def empty_cache():
        assert sys.exc_info()[0] is None
        assert buffers[0]() is None

    trainer._predict_transformed = predict
    with mock.patch("torch.cuda.empty_cache", side_effect=empty_cache) as cleanup:
        if fails_again:
            with pytest.raises(torch.cuda.OutOfMemoryError, match="second allocation"):
                trainer.predict(np.zeros((4, 2)), np.zeros(4), np.zeros((3, 2)))
            assert trainer.rng.rand() == draws[0]
        else:
            prediction = trainer.predict(np.zeros((4, 2)), np.zeros(4), np.zeros((3, 2)))
            assert prediction.shape == (3,)
    assert attempts == [(8, 2), (8, 2)]
    assert draws[0] == draws[1]
    cleanup.assert_called_once()


def test_training_oom_retry_terminates():
    import torch

    from autogluon.tabular.models.mitra.finetune import MitraFinetuneClassifier

    estimator = MitraFinetuneClassifier(max_samples_support=1, max_samples_query=1)
    estimator.y = np.array([0, 1])
    with (
        mock.patch(
            "autogluon.tabular.models.mitra.sklearn_interface.Tab2D.from_pretrained",
            side_effect=torch.cuda.OutOfMemoryError("test allocation"),
        ) as load,
        mock.patch("torch.cuda.empty_cache"),
    ):
        with pytest.raises(RuntimeError, match="Failed to train"):
            estimator._train_ensemble(None, None, None, None, "classification", 10)
    assert load.call_count == 1


def test_failed_fit_restores_torch_threads(checkpoint_factory, tmp_path):
    import torch

    X = pd.DataFrame(np.random.RandomState(0).normal(size=(16, 2)))
    y = pd.Series(np.arange(16) % 2)
    model = MitraModel(
        path=str(tmp_path / "mismatch"),
        problem_type="binary",
        hyperparameters={"hf_model": checkpoint_factory(version="1")},
    )
    original_threads = torch.get_num_threads()
    with pytest.raises(ValueError, match="model_version"):
        model.fit(X=X.iloc[:12], y=y.iloc[:12], X_val=X.iloc[12:], y_val=y.iloc[12:], num_cpus=2, num_gpus=0)
    assert torch.get_num_threads() == original_threads
