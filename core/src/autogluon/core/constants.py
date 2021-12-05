# Do not change these!
BINARY = 'binary'
MULTICLASS = 'multiclass'
REGRESSION = 'regression'
SOFTCLASS = 'softclass'  # classification with soft-target (rather than classes, labels are probabilities of each class).
QUANTILE = 'quantile'  # quantile regression (over multiple quantile levels, which are between 0.0 and 1.0)

PROBLEM_TYPES_CLASSIFICATION = [BINARY, MULTICLASS]
PROBLEM_TYPES_REGRESSION = [REGRESSION]
PROBLEM_TYPES = PROBLEM_TYPES_CLASSIFICATION + PROBLEM_TYPES_REGRESSION + [SOFTCLASS] + [QUANTILE]

PSEUDO_MODEL_SUFFIX = "_PSEUDO_{iter}"

REFIT_FULL_NAME = 'refit_single_full'  # stack-name used for refit_single_full (aka "compressed") models
REFIT_FULL_SUFFIX = "_FULL"  # suffix appended to model name for refit_single_full (aka "compressed") models

# AG_ARGS variables are key names in model hyperparameters to dictionaries of custom AutoGluon arguments.
# TODO: Have documentation for all AG_ARGS values
AG_ARGS = 'ag_args'  # Contains arguments to control model name, model priority, and the valid configurations which it can be used in.
AG_ARGS_FIT = 'ag_args_fit'  # Contains arguments that impact model training, such as early stopping rounds, #cores, #gpus, max time limit, max memory usage  # TODO
AG_ARGS_ENSEMBLE = 'ag_args_ensemble'  # Contains arguments that impact model ensembling, such as if an ensemble model is allowed to use the original features.  # TODO: v0.1 add to documentation

OBJECTIVES_TO_NORMALIZE = ['log_loss', 'pac_score', 'soft_log_loss']  # do not like predicted probabilities = 0

AUTO_WEIGHT = 'auto_weight'
BALANCE_WEIGHT = 'balance_weight'

# TODO: Add docs to dedicated page, or should it live in AbstractModel?
# TODO: How to reference correct version of docs?
# TODO: Add error in AG_ARGS if unknown key present
"""
AG_ARGS: Dictionary of customization options related to meta properties of the model such as its name, the order it is trained, and the problem types it is valid for.
    name: (str) The name of the model. This overrides AutoGluon's naming logic and all other name arguments if present.
    name_main: (str) The main name of the model. Example: 'RandomForest'.
    name_prefix: (str) Add a custom prefix to the model name. Unused by default.
    name_suffix: (str) Add a custom suffix to the model name. Unused by default.
    priority: (int) Determines the order in which the model is trained. Larger values result in the model being trained earlier. Default values range from 100 (RF) to 0 (custom), dictated by model type. If you want this model to be trained first, set priority = 999.
    problem_types: (list) List of valid problem types for the model. `problem_types=['binary']` will result in the model only being trained if `problem_type` is 'binary'.
    disable_in_hpo: (bool) If True, the model will only be trained if `hyperparameter_tune=False`.
    valid_stacker: (bool) If False, the model will not be trained as a level 1 or higher stacker model.
    valid_base: (bool) If False, the model will not be trained as a level 0 (base) model.
"""
