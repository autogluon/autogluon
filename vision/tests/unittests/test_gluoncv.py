from autogluon.vision._gluoncv import ImageClassification
from autogluon.vision._gluoncv import ObjectDetection
import autogluon.core as ag

IMAGE_CLASS_DATASET, _, IMAGE_CLASS_TEST = ImageClassification.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
OBJECT_DETCTION_DATASET = ObjectDetection.Dataset.from_voc('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip')
OBJECT_DETECTION_TRAIN, OBJECT_DETECTION_VAL, OBJECT_DETECTION_TEST = OBJECT_DETCTION_DATASET.random_split(val_size=0.3, test_size=0.2)


def test_image_classification():
    task = ImageClassification({'model': 'resnet18_v1', 'num_trials': 1, 'epochs': 1, 'batch_size': 8})
    classifier = task.fit(IMAGE_CLASS_DATASET)
    assert task.fit_summary().get('valid_acc', 0) > 0
    test_result = classifier.predict(IMAGE_CLASS_TEST)


def test_image_classification_custom_net():
    from gluoncv.model_zoo import get_model
    net = get_model('resnet18_v1')
    task = ImageClassification({'num_trials': 1, 'epochs': 1, 'custom_net': net, 'batch_size': 8})
    classifier = task.fit(IMAGE_CLASS_DATASET)
    assert task.fit_summary().get('valid_acc', 0) > 0
    test_result = classifier.predict(IMAGE_CLASS_TEST)


def test_object_detection_estimator():
    task = ObjectDetection({'num_trials': 1, 'epochs': 1, 'batch_size': 4})
    detector = task.fit(OBJECT_DETECTION_TRAIN)
    assert task.fit_summary().get('valid_map', 0) > 0
    test_result = detector.predict(OBJECT_DETECTION_TEST)


def test_object_detection_estimator_transfer():
    task = ObjectDetection({'num_trials': 1, 'epochs': 1, 'transfer': ag.Categorical('yolo3_darknet53_coco', 'ssd_512_resnet50_v1_voc'), 'estimator': 'ssd', 'batch_size': 4})
    detector = task.fit(OBJECT_DETECTION_TRAIN)
    assert task.fit_summary().get('valid_map', 0) > 0
    test_result = detector.predict(OBJECT_DETECTION_TEST)
