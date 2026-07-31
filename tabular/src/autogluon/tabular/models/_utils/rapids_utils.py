from typing import Dict

from autogluon.common.utils.resource_utils import ResourceManager


class RapidsModelMixin:
    """Mixin class for methods reused across RAPIDS models"""

    _default_ag_args_ensemble_extra = {"use_child_oof": False, "fold_fitting_strategy": "sequential_local"}
    minimum_num_gpus = 1
    gpu_required = True

    # FIXME: Efficient OOF doesn't work in RAPIDS

    def _get_default_resources(self):
        num_cpus, _ = super()._get_default_resources()
        num_gpus = min(
            ResourceManager.get_gpu_count_torch(), 1
        )  # Use single gpu training by default. Consider revising it later.
        return num_cpus, num_gpus

    def _more_tags(self):
        return {"valid_oof": False}
