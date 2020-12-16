from autogluon.tabular.models.text_prediction.text_prediction_v1_model import TextPredictionV1Model


def test_text_prediction_v1_sts(fit_helper):
    fit_args = dict(
        hyperparameters={TextPredictionV1Model: {}},
    )
    dataset_name = 'sts'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)
