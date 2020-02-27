# Do not change these!
BINARY = 'binary'
MULTICLASS = 'multiclass'
REGRESSION = 'regression'
SOFTCLASS = 'softclass' # classification with soft-target (rather than classes, labels are probabilities of each class).

PROBLEM_TYPES_CLASSIFICATION = [BINARY, MULTICLASS]
PROBLEM_TYPES_REGRESSION = [REGRESSION]
PROBLEM_TYPES = PROBLEM_TYPES_CLASSIFICATION + PROBLEM_TYPES_REGRESSION + [SOFTCLASS]

REFIT_FULL_NAME = 'refit_single_full'  # stack-name used for refit_single_full (aka "compressed") models
REFIT_FULL_SUFFIX = "_FULL"  # suffix appended to model name for refit_single_full (aka "compressed") models
