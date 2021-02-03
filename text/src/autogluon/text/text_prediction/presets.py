__all__ = ['ag_text_presets', 'list_presets']

import copy
import logging
from autogluon_contrib_nlp.utils.registry import Registry
from autogluon.core import space

logger = logging.getLogger(__name__)  # return root logger

ag_text_presets = Registry('ag_text_presets')


def list_presets():
    """List the presets available in AutoGluon-Text"""
    return ag_text_presets.list_keys()


@ag_text_presets.register()
def default() -> dict:
    """The default hyperparameters.

    It will have a version key and a list of candidate models.
    Each model has its own search space inside.
    """
    ret = {
        'version': 1,                     # Version of TextPrediction Model
        'models': {
            'MultimodalTextModel': {
                'backend': 'gluonnlp_v0',
                'search_space': {
                    'model.backbone.name': 'google_electra_base',
                    'optimization.batch_size': 128,
                    'optimization.per_device_batch_size': 8,
                    'optimization.num_train_epochs': 10,
                    'optimization.lr': space.Categorical(5E-5),
                    'optimization.wd': 1E-4,
                    'optimization.layerwise_lr_decay': 0.8
                }
            },
        },
        'misc': {
            'holdout_frac': None,  # If it is not provided, we will use the default strategy.
        },
        'hpo_params': {
            'search_strategy': 'random',   # Can be 'random', 'bayesopt', 'skopt',
                                           # 'hyperband', 'bayesopt_hyperband'
            'search_options': None,        # Extra kwargs passed to searcher
            'scheduler_options': None,     # Extra kwargs passed to scheduler
            'time_limits': None,           # The total time limit
            'num_trials': 1,               # The number of trials
        },
        'seed': None,                      # The seed value
    }
    return ret


@ag_text_presets.register()
def no_hpo() -> dict:
    """The default hyperparameters without HPO"""
    cfg = default()
    cfg['hpo_params']['num_trials'] = 1
    return cfg


@ag_text_presets.register()
def electra_small_no_hpo() -> dict:
    """The default search space that uses ELECTRA Small as the backbone."""
    cfg = no_hpo()
    cfg['models']['MultimodalTextModel']['search_space']['model.backbone.name'] \
        = 'google_electra_small'
    cfg['models']['MultimodalTextModel']['search_space'][
        'optimization.per_device_batch_size'] = 16
    return cfg


@ag_text_presets.register()
def electra_base_no_hpo() -> dict:
    """The default search space that uses ELECTRA Base as the backbone"""
    cfg = no_hpo()
    cfg['models']['MultimodalTextModel']['search_space']['model.backbone.name'] \
        = 'google_electra_base'
    cfg['models']['MultimodalTextModel']['search_space'][
        'optimization.per_device_batch_size'] = 8
    return cfg


@ag_text_presets.register()
def electra_large_no_hpo() -> dict:
    """The default search space that uses ELECTRA Base as the backbone."""
    cfg = no_hpo()
    cfg['models']['MultimodalTextModel']['search_space']['model.backbone.name'] \
        = 'google_electra_large'
    cfg['models']['MultimodalTextModel']['search_space'][
        'optimization.per_device_batch_size'] = 4
    return cfg


@ag_text_presets.register()
def roberta_base_no_hpo() -> dict:
    """The default search space that use ALBERT Base as the backbone."""
    cfg = no_hpo()
    cfg['models']['MultimodalTextModel']['search_space']['model.backbone.name'] \
        = 'fairseq_roberta_base'
    cfg['models']['MultimodalTextModel']['search_space']['optimization.per_device_batch_size'] = 8
    cfg['models']['MultimodalTextModel']['search_space']['optimization.layerwise_lr_decay'] = 0.8
    cfg['models']['MultimodalTextModel']['search_space']['model.network.text_net.use_segment_id'] = False
    return cfg


@ag_text_presets.register()
def multi_cased_bert_base_no_hpo() -> dict:
    """The default search space that use Multi-lingual BERT-Base as the backbone."""
    cfg = no_hpo()
    cfg['models']['MultimodalTextModel']['search_space']['model.backbone.name'] \
        = 'google_multi_cased_bert_base'
    cfg['models']['MultimodalTextModel']['search_space'][
        'optimization.per_device_batch_size'] = 8
    return cfg


def merge_params(base_params, partial_params=None):
    """Merge a partial change to the base configuration.

    Parameters
    ----------
    base_params
        The base parameters
    partial_params
        The partial parameters

    Returns
    -------
    final_params
        The final parameters
    """
    if partial_params is None:
        return base_params
    elif base_params is None:
        return partial_params
    else:
        if not isinstance(partial_params, dict):
            return partial_params
        assert isinstance(base_params, dict)
        final_params = copy.deepcopy(base_params)
        for key in partial_params:
            if key in base_params:
                final_params[key] = merge_params(base_params[key], partial_params[key])
            else:
                final_params[key] = partial_params[key]
        return final_params
