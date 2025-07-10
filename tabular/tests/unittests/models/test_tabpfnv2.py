from autogluon.tabular.models.tabpfnv2.tabpfnv2_model import TabPFNV2Model
from autogluon.tabular.testing import FitHelper

toy_model_params = {}


def test_tabpfnv2():
    model_cls = TabPFNV2Model
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)


def test_tabpfnv2_kditransform():
    model_cls = TabPFNV2Model
    model_hyperparameters = {
        "n_estimators": 1,
        # Check custom KDITransformer
        "inference_config/PREPROCESS_TRANSFORMS": [
            {
                "append_original": True,
                "categorical_name": "none",
                "global_transformer_name": None,
                "name": "kdi",
                "subsample_features": -1,
            }
        ],
    }

    FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
