from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from autogluon.tabular.models.lgb.lgb_model import LGBModel
from autogluon.tabular.testing import FitHelper, ModelFitHelper


def test_bagged_predict_children():
    fit_args = dict(k_fold=3)
    dataset_name = "toy_binary"
    model = BaggedEnsembleModel(
        model_base=LGBModel,
        model_base_kwargs=dict(hyperparameters=dict(num_boost_round=10)),  # Speed up run
        hyperparameters={"fold_fitting_strategy": "sequential_local"},  # Speed up run
    )
    ModelFitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        model=model,
        fit_args=fit_args,
        check_predict_children=True,
    )


# TODO: Test num_gpus>0
def test_resource_constraints():
    """
    Verify that num_cpus and num_gpus are respected when specified in the fit call.
    Also verifies that constraints are respected for weighted ensemble models and during refit_full.
    Also verifies that specifying num_cpus > max_cpus will use max_cpus.
    """
    num_gpus = 0

    max_cpus = ResourceManager.get_cpu_count()

    num_cpus_to_check = [1, 3]
    num_cpus_to_check = [c for c in num_cpus_to_check if c < max_cpus] + [max_cpus] + [max_cpus + 1000]

    for num_cpus in num_cpus_to_check:
        # Force DummyModel to raise an exception when fit.
        fit_args = dict(
            hyperparameters={"GBM": {"num_boost_round": 1}},
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            refit_full=True,
        )

        dataset_name = "toy_binary"
        predictor = FitHelper.fit_and_validate_dataset(
            dataset_name=dataset_name,
            fit_args=fit_args,
            expected_model_count=4,
            refit_full=False,
            delete_directory=False,
        )

        info = predictor.info()

        for model in predictor.model_names():
            if num_cpus <= max_cpus:
                assert info["model_info"][model]["num_cpus"] == num_cpus
            else:
                assert info["model_info"][model]["num_cpus"] == max_cpus  # Use max_cpus if num_cpus>max_cpus
            assert info["model_info"][model]["num_gpus"] == num_gpus
