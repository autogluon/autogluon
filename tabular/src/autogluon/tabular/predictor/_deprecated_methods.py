from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from autogluon.common.utils import Deprecated


class TabularPredictorDeprecatedMixin:
    """Contains deprecated methods from TabularPredictor that shouldn't show up in API documentation."""

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="persist")
    def persist_models(self, *args, **kwargs) -> List[str]:
        """Deprecated method. Use `persist` instead."""
        return self.persist(*args, **kwargs)

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="unpersist")
    def unpersist_models(self, *args, **kwargs) -> List[str]:
        """Deprecated method. Use `unpersist` instead."""
        return self.unpersist(*args, **kwargs)

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="model_names")
    def get_model_names(self, *args, **kwargs) -> List[str]:
        """Deprecated method. Use `model_names` instead."""
        return self.model_names(*args, **kwargs)

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="model_best")
    def get_model_best(self) -> str:
        """Deprecated method. Use `model_best` instead."""
        return self.model_best

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="predict_from_proba")
    def get_pred_from_proba(self, *args, **kwargs) -> pd.Series | np.array:
        """Deprecated method. Use `predict_from_proba` instead."""
        return self.predict_from_proba(*args, **kwargs)

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="model_refit_map")
    def get_model_full_dict(self, *args, **kwargs) -> Dict[str, str]:
        """Deprecated method. Use `model_refit_map` instead."""
        return self.model_refit_map(*args, **kwargs)

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="predict_proba_oof")
    def get_oof_pred_proba(self, *args, **kwargs) -> pd.DataFrame | pd.Series:
        """Deprecated method. Use `predict_proba_oof` instead."""
        return self.predict_proba_oof(*args, **kwargs)

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="predict_oof")
    def get_oof_pred(self, *args, **kwargs) -> pd.Series:
        """Deprecated method. Use `predict_oof` instead."""
        return self.predict_oof(*args, **kwargs)

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="disk_usage_per_file")
    def get_size_disk_per_file(self, *args, **kwargs) -> pd.Series:
        """Deprecated method. Use `disk_usage_per_file` instead."""
        return self.disk_usage_per_file(*args, **kwargs)

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="disk_usage")
    def get_size_disk(self) -> int:
        """Deprecated method. Use `disk_usage` instead."""
        return self.disk_usage()

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="model_names(persisted=True)")
    def get_model_names_persisted(self) -> List[str]:
        """Deprecated method. Use `model_names(persisted=True)` instead."""
        return self.model_names(persisted=True)
