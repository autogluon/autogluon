import pytest


# FIXME: Post 0.1, refit_full by storing the iteration idx for reaching the best performance.
@pytest.mark.gpu
def test_text_prediction_v1_sts(fit_helper):
    pytest.skip("Temporary skip the unittest")
    fit_args = dict(
        hyperparameters={'AG_TEXT_NN': {
            "optimization.max_epochs": 1,
            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
        }},
    )
    dataset_name = 'sts'
    fit_helper.fit_and_validate_dataset(
        dataset_name=dataset_name, fit_args=fit_args, sample_size=1000,
        refit_full=True,
    )
