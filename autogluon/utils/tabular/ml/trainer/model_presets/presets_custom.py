
from ...constants import AG_ARGS
from ...models.lgb.hyperparameters.parameters import get_param_baseline_custom


# Returns list of models created by a custom name preset.
def get_preset_custom(name, problem_type, num_classes):
    if not isinstance(name, str):
        raise ValueError(f'Expected string value for custom model, but was given {name}')
    # Custom model
    if name == 'GBM':
        model = get_param_baseline_custom(problem_type, num_classes=num_classes)
        model[AG_ARGS] = dict(model_type='GBM', name_suffix='Custom', disable_in_hpo=True)
        return [model]
    else:
        raise ValueError(f'Unknown custom model preset: {name}')
