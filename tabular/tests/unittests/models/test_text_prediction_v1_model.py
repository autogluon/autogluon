
def test_text_prediction_v1_sts(fit_helper):
    fit_args = dict(
        hyperparameters={'TEXT_NN_V1': {}},
        num_gpus=1,
    )
    dataset_name = 'sts'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name,
                                        fit_args=fit_args,
                                        sample_size=100,
                                        refit_full=True)
