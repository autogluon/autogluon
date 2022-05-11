import pytest

from autogluon.core.space import Categorical
from autogluon.vision._gluoncv import ObjectDetection


def get_dataset(path):
    return ObjectDetection.Dataset.from_voc(path)


@pytest.mark.skip(reason="ObjectDetector is not stable to test, and fails due to transient errors occasionally.")
def test_object_detection_estimator():
    dataset = get_dataset('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip')
    train_data, val_data, test_data = dataset.random_split(val_size=0.3, test_size=0.2, random_state=0)
    task = ObjectDetection({'num_trials': 1, 'epochs': 1, 'batch_size': 4})
    detector = task.fit(train_data)
    assert task.fit_summary().get('valid_map', 0) > 0
    test_result = detector.predict(test_data)


@pytest.mark.skip(reason="ObjectDetector is not stable to test, and fails due to transient errors occasionally.")
def test_object_detection_estimator_transfer():
    dataset = get_dataset('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip')
    train_data, val_data, test_data = dataset.random_split(val_size=0.3, test_size=0.2, random_state=0)
    task = ObjectDetection({'num_trials': 1, 'epochs': 1, 'transfer': Categorical('yolo3_darknet53_coco', 'ssd_512_resnet50_v1_voc'), 'estimator': 'ssd', 'batch_size': 4})
    detector = task.fit(train_data)
    assert task.fit_summary().get('valid_map', 0) > 0
    test_result = detector.predict(test_data)
