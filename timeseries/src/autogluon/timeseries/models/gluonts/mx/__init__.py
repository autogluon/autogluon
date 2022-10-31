import autogluon.timeseries as agts

if not agts.MXNET_INSTALLED:
    raise ImportError(
        "The MXNet models in autogluon.timeseries depend on Apache MXNet v1.9 or greater (below v2.0)."
        "Please install a suitable version of MXNet in order to use these models, with "
        "`pip install mxnet==1.9` or a matching MXNet package for your CUDA driver if you are using "
        "a GPU. See the MXNet documentation for more info."
    )


from .models import (
    DeepARMXNetModel,
    GenericGluonTSMXNetModel,
    MQCNNMXNetModel,
    MQRNNMXNetModel,
    SimpleFeedForwardMXNetModel,
    TemporalFusionTransformerMXNetModel,
    TransformerMXNetModel,
)

MXNET_MODEL_NAMES = [
    "DeepARMXNetModel",
    "MQCNNMXNetModel",
    "MQRNNMXNetModel",
    "SimpleFeedForwardMXNetModel",
    "TemporalFusionTransformerMXNetModel",
    "TransformerMXNetModel",
]
__all__ = MXNET_MODEL_NAMES + ["GenericGluonTSMXNetModel"]
