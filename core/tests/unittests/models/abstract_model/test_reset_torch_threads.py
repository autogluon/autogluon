from contextlib import nullcontext

import pandas as pd
import pytest

from autogluon.core.models import AbstractModel
from autogluon.core.utils.exceptions import TimeLimitExceeded

torch = pytest.importorskip("torch")


class _SetThreadsModel(AbstractModel):
    def _fit(self, X, y, num_cpus=1, **kwargs):
        torch.set_num_threads(num_cpus)
        self.model = "fitted"

    @classmethod
    def _class_tags(cls):
        return {"reset_torch_threads": True}


class _SetThreadsTimeoutModel(_SetThreadsModel):
    def _fit(self, X, y, num_cpus=1, **kwargs):
        torch.set_num_threads(num_cpus)
        raise TimeLimitExceeded


@pytest.fixture
def torch_threads():
    threads_og = torch.get_num_threads()
    torch.set_num_threads(3)
    yield 3
    torch.set_num_threads(threads_og)


@pytest.mark.parametrize(
    "model_cls, expectation",
    [(_SetThreadsModel, nullcontext()), (_SetThreadsTimeoutModel, pytest.raises(TimeLimitExceeded))],
)
def test_fit_restores_torch_threads(model_cls, expectation, torch_threads, tmp_path):
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 1.0, 2.0, 3.0])
    model = model_cls(path=str(tmp_path), name="m", problem_type="regression", eval_metric="root_mean_squared_error")
    with expectation:
        model.fit(X=X, y=y, num_cpus=1)
    assert torch.get_num_threads() == torch_threads
