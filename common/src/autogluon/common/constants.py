"""Vocabulary shared across AutoGluon packages.

Problem types live here rather than in ``autogluon.core`` so that utilities which are
themselves shared -- such as ``autogluon.common.utils.cv_splitter`` and
``autogluon.common.utils.validation_structure`` -- can reason about them without
``autogluon.common`` depending on ``autogluon.core``. ``autogluon.core.constants``
re-exports them, so existing imports from there keep working.
"""

# Do not change these!
BINARY = "binary"
MULTICLASS = "multiclass"
REGRESSION = "regression"
SOFTCLASS = (
    "softclass"  # classification with soft-target (rather than classes, labels are probabilities of each class).
)
QUANTILE = "quantile"  # quantile regression (over multiple quantile levels, which are between 0.0 and 1.0)

PROBLEM_TYPES_CLASSIFICATION = [BINARY, MULTICLASS]
PROBLEM_TYPES_REGRESSION = [REGRESSION]
PROBLEM_TYPES = PROBLEM_TYPES_CLASSIFICATION + PROBLEM_TYPES_REGRESSION + [SOFTCLASS] + [QUANTILE]
