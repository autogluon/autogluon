import numpy as np
from typing import Dict, Optional

import logging
from .meanstd_acqfunc import MeanStdAcquisitionFunction, HeadWithGradient
from ..tuning_algorithms.base_classes import OutputSurrogateModel, SurrogateModel
from ..utils.density import get_quantiles


logger = logging.getLogger(__name__)


class EIAcquisitionFunction(MeanStdAcquisitionFunction):
    """
    Minus expected improvement acquisition function
    (minus because the convention is to always minimize acquisition functions)

    """
    def __init__(self, model: OutputSurrogateModel, active_metric: str = None,
                 jitter: float = 0.01):
        assert isinstance(model, SurrogateModel)
        super().__init__(model, active_metric)
        self.jitter = jitter

    def _head_needs_current_best(self) -> bool:
        return True

    def _compute_heads(self, output_to_mean_std: Dict[str, Dict[str, np.ndarray]],
                       current_best: Optional[np.ndarray]) -> np.ndarray:
        means, stds = self._extract_active_metric_stats(output_to_mean_std)
        assert current_best is not None

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, means, stds)
        return (-stds) * (u * Phi + phi)

    def _compute_head_and_gradient(self, output_to_mean_std: Dict[str, Dict[str, np.ndarray]],
                                   current_best: Optional[np.ndarray]) -> HeadWithGradient:
        mean, std = self._extract_active_metric_stats(output_to_mean_std)
        assert current_best is not None

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, mean, std)
        f_acqu = std * (u * Phi + phi)
        return HeadWithGradient(
            hvals=-f_acqu,
            dh_dmean={self.active_metric: Phi},
            dh_dstd={self.active_metric: -phi})
