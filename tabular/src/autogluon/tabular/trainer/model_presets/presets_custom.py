from autogluon.core.constants import AG_ARGS

from ...models.lgb.hyperparameters.parameters import get_param_baseline_custom


# Returns list of models created by a custom name preset.
def get_preset_custom(name, problem_type):
    if not isinstance(name, str):
        raise ValueError(f"Expected string value for custom model, but was given {name}")
    # Custom model
    if name == "GBMLarge":
        model = get_param_baseline_custom(problem_type)
        model[AG_ARGS] = dict(model_type="GBM", name_suffix="Large", hyperparameter_tune_kwargs=None, priority=0)
        return [model]
    else:
        raise ValueError(f"Unknown custom model preset: {name}")
