from autogluon.vision._gluoncv import ImageClassification

IMAGE_CLASS_DATASET, _, IMAGE_CLASS_TEST = ImageClassification.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')


def test_torch_image_classification():
    task = ImageClassification({'model': 'resnet18', 'num_trials': 1, 'epochs': 1, 'batch_size': 8})
    classifier = task.fit(IMAGE_CLASS_DATASET)
    assert task.fit_summary().get('valid_acc', 0) > 0
    test_result = classifier.predict(IMAGE_CLASS_TEST)


def test_torch_image_classification_custom_net():
    from timm import create_model
    import torch.nn as nn
    net = create_model('resnet18')
    net.fc = nn.Linear(512, 4)
    task = ImageClassification({'num_trials': 1, 'epochs': 1, 'custom_net': net, 'batch_size': 8})
    classifier = task.fit(IMAGE_CLASS_DATASET)
    assert task.fit_summary().get('valid_acc', 0) > 0
    test_result = classifier.predict(IMAGE_CLASS_TEST)
