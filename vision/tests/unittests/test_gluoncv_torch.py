from autogluon.vision._gluoncv import ImageClassification


def get_dataset(path):
    train_data, _, test_data = ImageClassification.Dataset.from_folders(path)
    return train_data, test_data


def test_torch_image_classification():
    train_data, test_data = get_dataset('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    task = ImageClassification({'model': 'resnet18', 'num_trials': 1, 'epochs': 1, 'batch_size': 8})
    classifier = task.fit(train_data)
    assert task.fit_summary().get('valid_acc', 0) > 0
    test_result = classifier.predict(test_data)


def test_torch_image_classification_custom_net():
    train_data, test_data = get_dataset('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    from timm import create_model
    import torch.nn as nn
    net = create_model('resnet18')
    net.fc = nn.Linear(512, 4)
    task = ImageClassification({'num_trials': 1, 'epochs': 1, 'custom_net': net, 'batch_size': 8})
    classifier = task.fit(train_data)
    assert task.fit_summary().get('valid_acc', 0) > 0
    test_result = classifier.predict(test_data)
