from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from autogluon.tabular.models.lgb.lgb_model import LGBModel


def test_bagged_predict_children(model_fit_helper):
    fit_args = dict(k_fold=3)
    dataset_name = "adult"
    model = BaggedEnsembleModel(
        model_base=LGBModel,
        model_base_kwargs=dict(hyperparameters=dict(num_boost_round=10)),  # Speed up run
        hyperparameters={"fold_fitting_strategy": "sequential_local"},  # Speed up run
    )
    model_fit_helper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        model=model,
        fit_args=fit_args,
        check_predict_children=True,
        sample_size=100,
    )
