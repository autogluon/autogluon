import pytest

from autogluon.vision._gluoncv import ImageClassification


def get_dataset(path):
    train_data, _, test_data = ImageClassification.Dataset.from_folders(path)
    return train_data, test_data


@pytest.mark.skip(reason="ImagePredictor training with MXNet is not stable and may occasionally error. Skipping as MXNet backend is deprecated as of v0.4.0.")
def test_image_classification():
    train_data, test_data = get_dataset('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    task = ImageClassification({'model': 'resnet18_v1', 'num_trials': 1, 'epochs': 1, 'batch_size': 8})
    classifier = task.fit(train_data)
    assert task.fit_summary().get('valid_acc', 0) > 0
    test_result = classifier.predict(test_data)


@pytest.mark.skip(reason="ImagePredictor training with MXNet is not stable and may occasionally error. Skipping as MXNet backend is deprecated as of v0.4.0.")
def test_image_classification_custom_net():
    train_data, test_data = get_dataset('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    from gluoncv.model_zoo import get_model
    net = get_model('resnet18_v1')
    task = ImageClassification({'num_trials': 1, 'epochs': 1, 'custom_net': net, 'batch_size': 8})
    classifier = task.fit(train_data)
    assert task.fit_summary().get('valid_acc', 0) > 0
    test_result = classifier.predict(test_data)
