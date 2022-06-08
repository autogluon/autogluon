import pytest

try:
    import mxnet as mx
except ImportError:
    pytest.skip("MXNet is not installed. Skip this test.", allow_module_level=True)

from mxnet.util import use_np
import numpy as np
from autogluon_contrib_nlp.models import get_backbone
from autogluon.text.text_prediction.mx.modules import MultiModalWithPretrainedTextNN


@use_np
@pytest.mark.parametrize('num_text_features,num_categorical_features,num_numerical_features',
                         [(3, 4, 1),
                          (1, 0, 0),
                          (1, 4, 0),
                          (2, 0, 1)])
@pytest.mark.parametrize('aggregate_categorical', [False, True])
@pytest.mark.parametrize('agg_type', ['mean', 'concat', 'max',
                                      'attention', 'attention_token'])
@pytest.mark.parametrize('out_shape', [2])
def test_multimodal_with_pretrained_text_nn(num_text_features,
                                            num_categorical_features,
                                            num_numerical_features,
                                            aggregate_categorical,
                                            agg_type,
                                            out_shape):
    if agg_type == 'attention_token' and num_text_features != 1:
        pytest.skip('Not supported!')
    numerical_input_units = [32] * num_numerical_features
    num_categories = [np.random.randint(2, 10) for _ in range(num_categorical_features)]
    backbone_model_cls, backbone_cfg, tokenizer, backbone_params_path, _ = get_backbone(
        'google_electra_small')
    text_backbone = backbone_model_cls.from_cfg(backbone_cfg)
    cfg = MultiModalWithPretrainedTextNN.get_cfg()
    cfg.defrost()
    cfg.agg_net.agg_type = agg_type
    cfg.aggregate_categorical = aggregate_categorical
    cfg.freeze()
    net = MultiModalWithPretrainedTextNN(text_backbone=text_backbone,
                                         num_text_features=num_text_features,
                                         num_categorical_features=num_categorical_features,
                                         num_numerical_features=num_numerical_features,
                                         numerical_input_units=numerical_input_units,
                                         num_categories=num_categories,
                                         out_shape=out_shape,
                                         cfg=cfg)
    net.initialize_with_pretrained_backbone(backbone_params_path)
    net.hybridize()
    batch_size = 2
    seq_length = 5
    text_features = []
    for i in range(num_text_features):
        text_features.append((mx.np.random.randint(0, 100, (batch_size, seq_length)),
                              mx.np.random.randint(2, 5, (batch_size,)),
                              mx.np.zeros((batch_size, seq_length))))
    categorical_features = []
    for i, num_class in enumerate(num_categories):
        categorical_features.append(mx.np.random.randint(0, num_class, (batch_size,)))

    numerical_features = []
    for i, units in enumerate(numerical_input_units):
        numerical_features.append(mx.np.random.normal(0, 1, (batch_size, units)))
    gt = mx.np.random.normal(0, 1, (batch_size, out_shape))
    with mx.autograd.record():
        logits = net(text_features + categorical_features + numerical_features)
        loss = mx.np.square(logits - gt).mean()
        loss.backward()

