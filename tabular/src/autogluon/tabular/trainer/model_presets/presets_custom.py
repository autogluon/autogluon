from autogluon.core.constants import AG_ARGS
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
    elif name == 'ELECTRA_BASE':
        return _get_preset_electra('google_electra_base', 'default_no_hpo', 'Base')
    elif name == 'ELECTRA_BASE_HPO':
        return _get_preset_electra('google_electra_base', 'default', 'BaseHPO')
    elif name == 'ELECTRA_LARGE':
        return _get_preset_electra('google_electra_large', 'default_no_hpo', 'Large')
    else:
        raise ValueError(f'Unknown custom model preset: {name}')


def _get_preset_electra(backbone, params_name, name_suffix):
    from autogluon.text.text_prediction.text_prediction import ag_text_prediction_params
    model = ag_text_prediction_params.create(params_name)
    model['models']['BertForTextPredictionBasic']['search_space']['model.backbone.name'] = backbone
    model['models']['BertForTextPredictionBasic']['search_space']['optimization.num_train_epochs'] = 10
    model[AG_ARGS] = dict(model_type='TEXT_NN_V1', name_suffix=name_suffix, disable_in_hpo=True)
    return [model]
