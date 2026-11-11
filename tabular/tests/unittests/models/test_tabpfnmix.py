import os
import shutil

import pytest

from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from autogluon.tabular.models.tabpfnmix.tabpfnmix_model import TabPFNMixModel


class _DummyTrainer:
    def __init__(self):
        self.model = None


class _DummyInternalModel:
    def __init__(self):
        self.trainer = _DummyTrainer()


def test_tabpfnmix_load_resets_paths_after_clone(tmp_path):
    torch = pytest.importorskip("torch")

    orig_dir = str(tmp_path / "orig")
    clone_dir = str(tmp_path / "clone")

    model = TabPFNMixModel(path=orig_dir, name="TabPFNMix")
    model.model = _DummyInternalModel()
    fake_tensor = torch.tensor([1.0, 2.0, 3.0])
    model.model.trainer.model = fake_tensor
    model.save()

    shutil.copytree(orig_dir, clone_dir)
    shutil.rmtree(orig_dir)

    clone_model_path = os.path.join(clone_dir, "TabPFNMix")
    loaded = TabPFNMixModel.load(clone_model_path)
    assert loaded.path == clone_model_path
    assert loaded.weights_path == os.path.join(clone_model_path, "model.pt")
    assert torch.equal(loaded.model.trainer.model, fake_tensor)


def test_bagged_tabpfnmix_load_child_resets_paths_after_clone(tmp_path):
    torch = pytest.importorskip("torch")

    bag_orig_dir = str(tmp_path / "bag_orig")
    bag_clone_dir = str(tmp_path / "bag_clone")

    bag = BaggedEnsembleModel(
        path=bag_orig_dir,
        name="BaggedTabPFNMix",
        model_base=TabPFNMixModel(name="TabPFNMix_base"),
    )
    child_model = TabPFNMixModel(path=bag.path, name="S1F1")
    child_model.model = _DummyInternalModel()
    fake_tensor = torch.tensor([4.0, 5.0, 6.0])
    child_model.model.trainer.model = fake_tensor
    child_model.save()

    bag.add_child("S1F1")
    bag.save()

    shutil.copytree(bag_orig_dir, bag_clone_dir)
    shutil.rmtree(bag_orig_dir)

    cloned_bag_path = os.path.join(bag_clone_dir, "BaggedTabPFNMix")
    loaded_bag = BaggedEnsembleModel.load(cloned_bag_path)
    child = loaded_bag.load_child("S1F1")
    assert child.path == os.path.join(cloned_bag_path, "S1F1")
    assert child.weights_path == os.path.join(cloned_bag_path, "S1F1", "model.pt")
    assert torch.equal(child.model.trainer.model, fake_tensor)
