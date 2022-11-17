from autogluon.common.utils.log_utils import _add_stream_handler

from .predictor import (
    TabularCloudPredictor,
    TextCloudPredictor,
    ImageCloudPredictor,
    MultiModalCloudPredictor,
)

_add_stream_handler()
