import pytest

from autogluon.vision import ObjectDetector as Task


@pytest.mark.skip(reason="ObjectDetector is not stable to test, and fails due to transient errors occasionally.")
def test_task():
    dataset = Task.Dataset.from_voc('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip')
    train_data, _, test_data = dataset.random_split(random_state=0)

    detector = Task()
    detector.fit(train_data, hyperparameters={'batch_size': 4, 'epochs': 5, 'early_stop_max_value': 0.2}, hyperparameter_tune_kwargs={'num_trials': 1})
    test_result = detector.predict(test_data)
    detector.save('detector.ag')
    detector2 = Task.load('detector.ag')
    fit_summary = detector2.fit_summary()
    test_map = detector2.evaluate(test_data)
    test_result2 = detector2.predict(test_data)
    assert test_result2.equals(test_result), f'{test_result2} != \n {test_result}'
    # to numpy
    test_result2 = detector2.predict(test_data, as_pandas=False)
