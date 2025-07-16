import logging

from autogluon.common.utils.try_import import try_import_rapids_cuml
from autogluon.core.constants import REGRESSION

from .._utils.rapids_utils import RapidsModelMixin
from .hyperparameters.parameters import get_param_baseline
from .lr_model import LinearModel

logger = logging.getLogger(__name__)


# FIXME: If rapids is installed, normal CPU LinearModel crashes.
class LinearRapidsModel(RapidsModelMixin, LinearModel):
    """
    RAPIDS Linear model : https://rapids.ai/start.html

    NOTE: This code is experimental, it is recommend to not use this unless you are a developer.
    This was tested on rapids-21.06 via:

    conda create -n rapids-21.06 -c rapidsai -c nvidia -c conda-forge rapids=21.06 python=3.8 cudatoolkit=11.2
    conda activate rapids-21.06
    pip install autogluon.tabular[all]
    """

    def _get_model_type(self):
        penalty = self.params.get("penalty", "L2")
        try_import_rapids_cuml()
        from cuml.linear_model import Lasso, LogisticRegression, Ridge

        if self.problem_type == REGRESSION:
            if penalty == "L2":
                model_type = Ridge
            elif penalty == "L1":
                model_type = Lasso
            else:
                raise AssertionError(f'Unknown value for penalty "{penalty}" - supported types are ["L1", "L2"]')
        else:
            model_type = LogisticRegression
        return model_type

    def _set_default_params(self):
        default_params = {"fit_intercept": True, "max_iter": 10000}
        if self.problem_type != REGRESSION:
            default_params.update({"solver": "qn"})
        default_params.update(get_param_baseline())
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X=X, **kwargs)
        if hasattr(X, 'toarray'):  # Check if it's a sparse matrix
            X = X.toarray()
        return X

    def _fit(self, X, y, **kwargs):
        """
        Custom fit method for RAPIDS cuML models that handles parameter compatibility
        and bypasses sklearn-specific incremental training approach.
        """
        # Preprocess data
        X = self.preprocess(X, is_train=True)
        if self.problem_type == 'binary':
            y = y.astype(int).values

        # Create cuML model with filtered parameters
        model_cls = self._get_model_type()

        # Comprehensive parameter filtering for cuML compatibility
        cuml_incompatible_params = {
            # AutoGluon-specific preprocessing parameters
            'vectorizer_dict_size', 'proc.ngram_range', 'proc.skew_threshold',
            'proc.impute_strategy', 'handle_text',
            # sklearn-specific parameters not supported by cuML
            'n_jobs', 'warm_start', 'multi_class', 'dual', 'intercept_scaling',
            'class_weight', 'random_state', 'verbose',
            # Parameters that need conversion or special handling
            'penalty', 'C'
        }

        # Filter out incompatible parameters
        filtered_params = {k: v for k, v in self.params.items()
                          if k not in cuml_incompatible_params}

        # Handle parameter conversions for cuML
        if self.problem_type == REGRESSION:
            # Convert sklearn's C parameter to cuML's alpha
            if 'C' in self.params:
                filtered_params['alpha'] = 1.0 / self.params['C']
        else:
            # For classification, keep C parameter
            if 'C' in self.params:
                filtered_params['C'] = self.params['C']

        # Create and fit cuML model - let cuML handle its own error messages
        self.model = model_cls(**filtered_params)
        self.model.fit(X, y)

        # Add missing sklearn-compatible attributes for AutoGluon compatibility
        self.model.n_iter_ = None  # cuML doesn't track iterations like sklearn
