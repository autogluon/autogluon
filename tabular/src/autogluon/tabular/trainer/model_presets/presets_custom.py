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
        from autogluon.text.text_prediction.text_prediction import ag_text_prediction_params
        text_backbone = 'google_electra_base'
        model = ag_text_prediction_params.create('default_no_hpo')
        model['models']['BertForTextPredictionBasic']['search_space']['model.backbone.name'] = text_backbone
        model['models']['BertForTextPredictionBasic']['search_space']['optimization.num_train_epochs'] = 10
        model[AG_ARGS] = dict(model_type='TEXT_NN_V1', name_suffix='Base', disable_in_hpo=True)
        return [model]
    elif name == 'ELECTRA_BASE_HPO':
        from autogluon.text.text_prediction.text_prediction import ag_text_prediction_params
        text_backbone = 'google_electra_base'
        model = ag_text_prediction_params.create('default')
        model['models']['BertForTextPredictionBasic']['search_space']['model.backbone.name'] = text_backbone
        model['models']['BertForTextPredictionBasic']['search_space']['optimization.num_train_epochs'] = 10
        model[AG_ARGS] = dict(model_type='TEXT_NN_V1', name_suffix='BaseHPO', disable_in_hpo=True)
        return [model]
    elif name == 'ELECTRA_LARGE':
        from autogluon.text.text_prediction.text_prediction import ag_text_prediction_params
        text_backbone = 'google_electra_large'
        model = ag_text_prediction_params.create('default_no_hpo')
        model['models']['BertForTextPredictionBasic']['search_space']['model.backbone.name'] = text_backbone
        model['models']['BertForTextPredictionBasic']['search_space']['optimization.num_train_epochs'] = 10
        model[AG_ARGS] = dict(model_type='TEXT_NN_V1', name_suffix='Large', disable_in_hpo=True)
        return [model]
    else:
        raise ValueError(f'Unknown custom model preset: {name}')
