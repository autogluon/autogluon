import logging


from autogluon.common.features.types import R_OBJECT, R_INT, R_FLOAT, R_CATEGORY, \
    S_TEXT_NGRAM, S_TEXT_AS_CATEGORY, S_TEXT_SPECIAL, S_IMAGE_PATH

from ..automm.automm_model import MultiModalPredictorModel

logger = logging.getLogger(__name__)


class TextPredictorModel(MultiModalPredictorModel):
    """MultimodalPredictor that doesn't use image features"""
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_INT, R_FLOAT, R_CATEGORY, R_OBJECT],
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_AS_CATEGORY, S_TEXT_SPECIAL, S_IMAGE_PATH],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
