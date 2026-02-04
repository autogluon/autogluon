from __future__ import annotations

from ..lgb.lgb_model import LGBModel
from .prep_mixin import ModelAgnosticPrepMixin


class PrepLGBModel(ModelAgnosticPrepMixin, LGBModel):
    ag_key = "GBM_PREP"
    ag_name = "LightGBMPrep"

    @classmethod
    def _estimate_memory_usage_static(cls, **kwargs) -> int:
        memory_usage = super()._estimate_memory_usage_static(**kwargs)
        # FIXME: 1.5 runs OOM on kddcup09_appetency fold 2 repeat 0 prep_LightGBM_r49_BAG_L1
        return memory_usage * 2.0  # FIXME: For some reason this underestimates mem usage without this

    @classmethod
    def _estimate_memory_usage_static_lite(cls, **kwargs) -> int:
        memory_usage = super()._estimate_memory_usage_static_lite(**kwargs)
        # FIXME: 1.5 runs OOM on kddcup09_appetency fold 2 repeat 0 prep_LightGBM_r49_BAG_L1
        return memory_usage * 2.0  # FIXME: For some reason this underestimates mem usage without this
