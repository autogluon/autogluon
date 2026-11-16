"""Sequential Mitra bags with a shared training deadline."""

import logging
import time

from autogluon.core.models.ensemble.fold_fitting_strategy import SequentialLocalFoldFittingStrategy
from autogluon.core.utils.exceptions import TimeLimitExceeded

logger = logging.getLogger(__name__)


class MitraSequentialLocalFoldFittingStrategy(SequentialLocalFoldFittingStrategy):
    """Keep the remaining-time allocation without extrapolating startup cost.

    Mitra may retry training with smaller contexts after an OOM. Later folds
    reuse the fitted context size, so multiplying the first fold's duration
    can reject a bag that would finish within its deadline.
    """

    def _get_fold_time_limit(self, fold_ctx):
        try:
            return super()._get_fold_time_limit(fold_ctx)
        except TimeLimitExceeded:
            time_left = self.time_limit - (time.time() - self.time_start)
            if time_left <= 0:
                raise
            logger.info("Reallocating %.1fs remaining across %d Mitra folds.", time_left, fold_ctx["folds_left"])
            return (time_left / fold_ctx["folds_left"]) * self.time_limit_fold_ratio

    def _predict_oof(self, fold_model, fold_ctx):
        # Score a completed fold without projecting its cost onto future folds.
        # The next fold still receives only its share of the remaining deadline.
        _, val_index = fold_ctx["fold"]
        X_val_fold = self.X.iloc[val_index, :]
        y_val_fold = self.y.iloc[val_index]
        y_pred_proba = fold_model.predict_proba(X_val_fold, record_time=True)
        fold_model.val_score = fold_model.score_with_y_pred_proba(y=y_val_fold, y_pred_proba=y_pred_proba)
        fold_model.reduce_memory_size(remove_fit=True, remove_info=False, requires_save=True)
        if not self.bagged_ensemble_model.save_bag_folds:
            fold_model.model = None
        return fold_model, y_pred_proba
