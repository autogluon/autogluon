"""Verification for the pre-Ampere precision fix (removed again before review, see the PR description).

On a CUDA GPU older than Ampere (e.g. T4), bfloat16 has no memory-efficient attention kernel, so Mitra's
attention falls back to the math kernel and the OOM handler halves max_samples_support during fit.
With float16 the default 8192-row support fits.
"""

import pytest
import torch
from sklearn.datasets import make_classification


@pytest.mark.gpu
def test_mitra_keeps_full_support_on_pre_ampere_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] >= 8:
        pytest.skip("needs a pre-Ampere CUDA GPU (e.g. T4)")
    from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier

    X, y = make_classification(n_samples=11500, n_features=10, n_informative=6, weights=[0.9], random_state=0)
    model = MitraClassifier(fine_tune=False, n_estimators=1, device="cuda").fit(X[:9500], y[:9500])
    assert model.trainers[0].cfg.hyperparams["max_samples_support"] == 8192
