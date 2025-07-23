import pytest

from autogluon.tabular.testing import FitHelper


# FIXME: Post 0.1, refit_full by storing the iteration idx for reaching the best performance.
@pytest.mark.gpu
def test_text_prediction_v1_sts():
    pytest.skip("Temporary skip the unittest")
    fit_args = dict(
        hyperparameters={
            "AG_TEXT_NN": {
                "optim.max_epochs": 1,
                "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
            }
        },
    )
    dataset_name = "sts"
    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        sample_size=100,
        refit_full=True,
    )
