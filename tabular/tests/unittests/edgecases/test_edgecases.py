
from autogluon.core.constants import BINARY
from autogluon.core.metrics import METRICS


def test_no_weighted_ensemble(fit_helper):
    """Tests that fit_weighted_ensemble=False works"""
    fit_args = dict(
        hyperparameters={'GBM': {}},
        fit_weighted_ensemble=False,
    )
    dataset_name = 'adult'
    extra_metrics = list(METRICS[BINARY])

    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name,
                                        fit_args=fit_args,
                                        extra_metrics=extra_metrics,
                                        expected_model_count=1)
