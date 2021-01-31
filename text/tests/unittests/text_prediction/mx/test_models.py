import mxnet as mx
from mxnet.util import use_np
import numpy as np
import pytest
from autogluon_contrib_nlp.models import get_backbone
from autogluon.text.text_prediction.mx.modules import MultiModalWithPretrainedTextNN
from autogluon.text.text_prediction.mx.models import infer_per_device_batch_size


@use_np
@pytest.mark.parametrize('num_categories,numerical_units,max_length',
                         [([20, 32, 2, 5, 4], 32, 256)])
@pytest.mark.parametrize('backbone',
                         ['google_electra_small'])
@pytest.mark.parametrize('agg_type', ['attention'])
@pytest.mark.parametrize('input_gating', [True])
@pytest.mark.parametrize('out_shape', [100])
def test_infer_per_device_batch_size(num_categories,
                                     numerical_units,
                                     max_length,
                                     backbone,
                                     agg_type,
                                     input_gating,
                                     out_shape):
    ctx = mx.gpu(0)
    backbone_model_cls, backbone_cfg, tokenizer, backbone_params_path, _ = get_backbone(backbone)
    text_backbone = backbone_model_cls.from_cfg(backbone_cfg)
    cfg = MultiModalWithPretrainedTextNN.get_cfg()
    cfg.defrost()
    cfg.agg_net.agg_type = agg_type
    cfg.agg_net.input_gating = input_gating
    cfg.freeze()
    net = MultiModalWithPretrainedTextNN(text_backbone=text_backbone,
                                         num_text_features=1,
                                         num_categorical_features=len(num_categories),
                                         num_numerical_features=1,
                                         numerical_input_units=numerical_units,
                                         num_categories=num_categories,
                                         out_shape=out_shape,
                                         cfg=cfg)
    net.initialize_with_pretrained_backbone(backbone_params_path, ctx=ctx)
    net.hybridize()
    per_device_batch_size = infer_per_device_batch_size(net, init_batch_size=4,
                                                        max_length=max_length,
                                                        num_categories=num_categories,
                                                        numerical_units=numerical_units,
                                                        ctx=ctx)
    assert per_device_batch_size >= 1
    print(per_device_batch_size)
