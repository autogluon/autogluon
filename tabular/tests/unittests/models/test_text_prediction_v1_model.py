
# FIXME: Post 0.1, refit_full by storing the iteration idx for reaching the best performance.
def test_text_prediction_v1_sts(fit_helper):
    fit_args = dict(
        hyperparameters={'AG_TEXT_NN': {}},
    )
    dataset_name = 'sts'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args, sample_size=100)
