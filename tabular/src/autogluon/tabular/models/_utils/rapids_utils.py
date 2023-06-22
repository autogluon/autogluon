from typing import Dict

from autogluon.common.utils.resource_utils import ResourceManager


class RapidsModelMixin:
    """Mixin class for methods re-used across RAPIDS models"""

    # FIXME: Efficient OOF doesn't work in RAPIDS
    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {"use_child_oof": False}
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _get_default_resources(self):
        num_cpus, _ = super()._get_default_resources()
        num_gpus = min(ResourceManager.get_gpu_count_torch(), 1)  # Use single gpu training by default. Consider revising it later.
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available=False) -> Dict[str, int]:
        return {
            "num_cpus": 1,
            "num_gpus": 1,
        }

    def _more_tags(self):
        return {"valid_oof": False}
