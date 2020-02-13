# Do not change these!
BINARY = 'binary'
MULTICLASS = 'multiclass'
REGRESSION = 'regression'
SOFTCLASS = 'softclass' # classification with soft-target (rather than classes, labels are probabilities of each class).

PROBLEM_TYPES_CLASSIFICATION = [BINARY, MULTICLASS]
PROBLEM_TYPES_REGRESSION = [REGRESSION]
PROBLEM_TYPES = PROBLEM_TYPES_CLASSIFICATION + PROBLEM_TYPES_REGRESSION + [SOFTCLASS]
