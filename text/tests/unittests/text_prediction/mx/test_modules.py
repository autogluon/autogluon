import mxnet as mx
from mxnet.util import use_np
import pytest
from autogluon.text.text_prediction.mx.modules import MultiModalWithPretrainedTextNN


@use_np
@pytest.mark.parametrize('num_text_features,num_categorical_features,num_numerical_features',
                         [(3, 4, 1),
                          (1, 0, 0),
                          (1, 4, 0),
                          (2, 0, 1)])
@pytest.mark.parametrize('agg_type', ['mean', 'concat', 'attention'])
@pytest.mark.parametrize('input_gating', [False, True])
def test_multimodal_with_pretrained_text_nn(num_text_features,
                                            num_categorical_features,
                                            num_numerical_features,
                                            agg_type,
                                            input_gating):
    pass
