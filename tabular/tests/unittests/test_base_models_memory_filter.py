from unittest import mock

import pandas as pd

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.tabular.trainer.auto_trainer import AutoTrainer


class _MockTrainer(AutoTrainer):
    def __init__(self, problem_type=MULTICLASS, classes=None, val_scores=None):
        self.problem_type = problem_type
        self.classes = classes if classes is not None else [f"c{i}" for i in range(100)]
        self._val_scores = val_scores or {}

    def get_models_attribute_dict(self, attribute="val_score", models=None):
        if attribute == "val_score":
            return {m: self._val_scores.get(m, 0.5) for m in (models or [])}
        return {}


def test_filter_base_models_via_memory_limit_no_pruning_when_plenty_of_memory(caplog):
    val_scores = {f"M{i}": float(i) for i in range(5)}
    trainer = _MockTrainer(problem_type=MULTICLASS, classes=[f"c{i}" for i in range(3)], val_scores=val_scores)
    base_models = [f"M{i}" for i in range(5)]
    X = pd.DataFrame({"feat": range(100)})

    with mock.patch.object(ResourceManager, "get_available_virtual_mem", return_value=16 * 1024**3):
        with caplog.at_level("WARNING"):
            retained = trainer._filter_base_models_via_memory_limit(base_models, X=X, max_memory_ratio=0.25)

    assert retained == base_models
    assert "Pruning" not in caplog.text


def test_filter_base_models_via_memory_limit_prunes_lowest_scoring_models(caplog):
    # S=100_000 rows, C=100 classes -> bytes_per_base_model = 100_000 * 100 * 4 = 40_000_000 B (40 MB)
    val_scores = {f"M{i}": float(i) for i in range(10)}  # M9 is best, M0 is worst
    trainer = _MockTrainer(problem_type=MULTICLASS, classes=[f"c{i}" for i in range(100)], val_scores=val_scores)
    base_models = [f"M{i}" for i in range(10)]
    X = pd.DataFrame({"feat": range(100_000)})

    # Set available memory such that 25% allows exactly 2 models (e.g., max_allowed = 100 MB -> 2 models * 40 MB = 80 MB)
    mock_available_mem = 400 * 10**6  # 400 MB -> 25% is 100 MB

    with mock.patch.object(ResourceManager, "get_available_virtual_mem", return_value=mock_available_mem):
        with caplog.at_level("WARNING"):
            retained = trainer._filter_base_models_via_memory_limit(base_models, X=X, max_memory_ratio=0.25)

    assert len(retained) == 2
    assert set(retained) == {"M9", "M8"}  # top 2 scores
    assert "Pruning 8 base models due to memory constraints (2 retained, max allowed: 2" in caplog.text


def test_filter_base_models_via_memory_limit_always_retains_at_least_one():
    val_scores = {"M0": 0.1, "M1": 0.9}
    trainer = _MockTrainer(problem_type=BINARY, val_scores=val_scores)
    base_models = ["M0", "M1"]
    X = pd.DataFrame({"feat": range(10_000)})

    # Mock virtually 0 available memory
    with mock.patch.object(ResourceManager, "get_available_virtual_mem", return_value=10):
        retained = trainer._filter_base_models_via_memory_limit(base_models, X=X, max_memory_ratio=0.25)

    assert len(retained) == 1
    assert retained == ["M1"]  # Best model retained even under extreme memory pressure


def test_filter_base_models_via_memory_limit_empty():
    trainer = _MockTrainer(problem_type=REGRESSION)
    assert trainer._filter_base_models_via_memory_limit([], X=None) == []
    assert trainer._filter_base_models_via_memory_limit(["M1"], X=None) == ["M1"]
    assert trainer._filter_base_models_via_memory_limit(["M1"], X=pd.DataFrame()) == ["M1"]
