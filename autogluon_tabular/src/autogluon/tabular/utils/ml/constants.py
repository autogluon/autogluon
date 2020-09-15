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

# AG_ARGS variables are key names in model hyperparameters to dictionaries of custom AutoGluon arguments.
AG_ARGS = 'AG_args'  # Contains arguments to control model name, model priority, and the valid configurations which it can be used in.
AG_ARGS_FIT = 'AG_args_fit'  # Contains arguments that impact model training, such as early stopping rounds, #cores, #gpus, max time limit, max memory usage  # TODO

OBJECTIVES_TO_NORMALIZE = ['log_loss', 'pac_score', 'soft_log_loss']  # do not like predicted probabilities = 0
