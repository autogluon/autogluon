import mxnet as mx
import pytest

from autogluon.extra.model_zoo import get_model

x = mx.nd.random.uniform(shape=(1, 3, 224, 224))


@pytest.mark.parametrize("model_name", [
    'standford_dog_resnet152_v1', 'standford_dog_resnext101_64x4d',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
    'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
    'efficientnet_b6', 'efficientnet_b7'
])
def test_image_classification_models(model_name):
    # get the model
    net = get_model(model_name, pretrained=True)
    # test inference
    _ = net(x)
