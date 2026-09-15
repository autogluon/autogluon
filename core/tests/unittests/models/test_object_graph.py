"""The object-graph walkers behind the shared-weights mixin, on an estimator shaped like tabpfn's with a many-class wrapper."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from autogluon.core.models.abstract import _object_graph as g

torch = pytest.importorskip("torch")


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)


class _Specs:
    """A container payload: network, criterion module, config."""

    def __init__(self):
        self.model = _Net()
        self.criterion = _Net()
        self.config = {"a": 1}


class _Cache:
    def __init__(self, model):
        self._models = {torch.device("cpu"): model}


class _Executor:
    def __init__(self, model):
        self.model_caches = [_Cache(model)]


class _Estimator:
    def __init__(self, specs):
        self.model_path = specs
        self.device = "cpu"
        self.devices_ = (torch.device("cpu"),)
        self.models_ = [specs.model]
        self.executor_ = _Executor(specs.model)
        self.znorm_ = specs.criterion
        self.raw_ = torch.nn.Linear(1, 1)
        self.V = torch.zeros(3)
        self.arr = np.zeros(2)
        self.inference_config_ = {"COL": {"device": torch.device("cpu")}}


class _ManyClass:
    def __init__(self, base):
        self.estimator = base
        self.estimators_ = [_Estimator(base.model_path) for _ in range(2)]


@pytest.fixture
def payload():
    specs = _Specs()
    g.mark_identity_under_deepcopy(specs)
    return specs


def test_components_and_identity_under_deepcopy(payload):
    assert copy.deepcopy(payload) is payload
    components = g.shared_components(payload)
    assert components[0] is payload and components[1] is payload.model and components[2] is payload.criterion
    assert len(components) == 3
    net = _Net()
    g.mark_identity_under_deepcopy(net)
    assert copy.deepcopy(net) is not net, "a bare module keeps deep-copying (libraries make private copies on purpose)"


def test_find_detach_attach_round_trip(payload):
    components = g.shared_components(payload)
    root = _ManyClass(_Estimator(payload))
    found = g.find_shared(root, components)
    leaves = {p[-1] for p, _ in found}
    assert ("attr", "model_path") in leaves and ("attr", "znorm_") in leaves and ("index", 0) in leaves
    assert ("key", torch.device("cpu")) in leaves, "the per-device cache entry was not found"
    assert len(found) == 12

    detached, paths, moved = g.detach_shared(root, components)
    assert len(paths) == len(found)
    assert moved == [], "every tensor already lived on the CPU"
    assert root.estimator.models_[0] is payload.model, "the live estimator was modified"
    assert detached.estimator.models_[0] is None and detached.estimator.model_path is None
    assert detached.estimators_[1].executor_.model_caches[0]._models[torch.device("cpu")] is None
    assert detached.estimators_[0].znorm_ is None
    assert isinstance(detached.estimators_[0].raw_, torch.nn.Module), "an owned module stays"
    assert detached.estimators_[0].arr is root.estimators_[0].arr, "data leaves are shared, not copied"
    blob = pickle.dumps(vars(detached))
    assert b"lin.weight" not in blob or len(blob) < 6000

    restored = _ManyClass.__new__(_ManyClass)
    restored.__dict__.update(pickle.loads(blob))
    other = _Specs()
    restored = g.attach_shared(restored, paths, g.shared_components(other))
    assert restored.estimator.models_[0] is other.model
    assert restored.estimators_[1].znorm_ is other.criterion
    assert restored.estimators_[0].model_path is other
    assert restored.estimators_[1].executor_.model_caches[0]._models[torch.device("cpu")] is other.model


def test_detach_drops_bound_method_shadows_and_moves_tensors_to_cpu(payload):
    est = _Estimator(payload)
    est.__dict__["_load_model"] = est.__init__
    detached, _, _ = g.detach_shared(est, g.shared_components(payload))
    assert "_load_model" not in vars(detached)
    assert all(t.device.type == "cpu" for _, t in [(None, detached.V)])
    assert g.tensors_off_device(detached, "cpu", skip=g.shared_components(payload)) == []


def test_rewrite_devices_rewrites_fields_keys_and_skips_the_payload(payload):
    components = g.shared_components(payload)
    est = _Estimator(payload)
    est.device = "cuda:0"
    est.devices_ = (torch.device("cuda"),)
    est.inference_config_["COL"]["device"] = torch.device("cuda")
    est.executor_.model_caches[0]._models = {torch.device("cuda"): payload.model}
    g.rewrite_devices(est, "cuda", "cpu", skip=components)
    assert est.device == "cpu"
    assert est.devices_ == (torch.device("cpu"),)
    assert est.inference_config_["COL"]["device"] == torch.device("cpu")
    assert list(est.executor_.model_caches[0]._models) == [torch.device("cpu")]
    assert est.models_[0] is payload.model, "the shared module was replaced"
    assert g.rewrite_devices(est, "cpu", "cpu") is est


def test_assign_rebuilds_tuples():
    root = {"a": (1, [2, 3])}
    g.assign(root, (("key", "a"), ("index", 1), ("index", 0)), 9)
    assert root == {"a": (1, [9, 3])}
    root = g.assign(root, (("key", "a"), ("index", 0)), 7)
    assert root["a"] == (7, [9, 3])
    assert g.get_at(root, (("key", "a"), ("index", 1), ("index", 0))) == 9


def test_payload_device_and_modules_on_dicts_and_containers(payload):
    assert g.payload_device(payload) == "cpu"
    assert g.payload_device({"w": torch.zeros(1)}) == "cpu"
    assert g.payload_modules(payload) == [payload.model, payload.criterion]
    g.check_payload_device(payload, "cpu")
    with pytest.raises(RuntimeError):
        g.check_payload_device(payload, "cuda")
