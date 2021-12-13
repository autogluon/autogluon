__all__ = ['ag_text_presets', 'list_presets']

import copy
import logging
import functools
from autogluon_contrib_nlp.utils.registry import Registry
from autogluon.core import space

logger = logging.getLogger(__name__)  # return root logger

# TODO, Consider to move the registry to core
ag_text_presets = Registry('ag_text_presets')


def list_presets():
    """List the presets available in AutoGluon-Text"""
    simple_presets = ['default', 'lower_quality_fast_train',
                      'medium_quality_faster_train', 'best_quality']
    return {'simple_presets': simple_presets,
            'advanced_presets': [key for key in ag_text_presets.list_keys()
                        if key not in simple_presets]}


def base() -> dict:
    """The default hyperparameters.

    It will have a version key and a list of candidate models.
    Each model has its own search space inside.
    """
    ret = {
        'models': {
            'MultimodalTextModel': {
                'backend': 'gluonnlp_v0',
                'search_space': {
                    'model.backbone.name': 'google_electra_base',
                    'optimization.batch_size': 128,
                    'optimization.per_device_batch_size': 4,
                    'optimization.num_train_epochs': 10,
                    'optimization.lr': space.Categorical(1E-4),
                    'optimization.wd': 1E-4,
                    'optimization.layerwise_lr_decay': 0.8
                }
            },
        },
        'tune_kwargs': {                                  # Same as the hyperparameter_tune_kwargs in AutoGluon Tabular.
            'search_strategy': 'local',                   # Can be 'random', 'bayesopt', 'skopt',
                                                          # 'hyperband', 'bayesopt_hyperband'
            'searcher': 'local_random',
            'search_options': None,                       # Extra kwargs passed to searcher
            'scheduler_options': None,                    # Extra kwargs passed to scheduler
            'num_trials': 1,                              # The number of trials
        },
    }
    return ret


def apply_average_nbest(cfg, nbest=3):
    """Apply the average checkpoint trick to the basic configuration.

    Parameters
    ----------
    cfg
        The basic configuration
    nbest
        The number of best checkpoints to average

    Returns
    -------
    new_cfg
        The new configuration
    """
    new_cfg = copy.deepcopy(cfg)
    search_space = new_cfg['models']['MultimodalTextModel']['search_space']
    search_space['model.use_avg_nbest'] = True
    search_space['optimization.nbest'] = nbest
    return new_cfg


def apply_fusion_strategy(cfg, strategy='fuse_late'):
    """Apply the specific fusion strategy to a basic config

    Parameters
    ----------
    cfg
        The basic configuration
    strategy
        The type of the fusion strategy, can be
        - fuse_late:
            Use separate networks for extracting features of text, numerical and categorical data.
        - fuse_early:
            Keep the token embeddings and use an additional transformer to fuse the information.
        - all_text:
            Convert all categorical and numerical data to text and then use the text network

    Returns
    -------
    new_cfg
        The new configuration.
    """
    new_cfg = copy.deepcopy(cfg)
    search_space = new_cfg['models']['MultimodalTextModel']['search_space']
    if strategy == 'fuse_late':
        search_space['model.network.agg_net.agg_type'] = 'concat'
        search_space['model.network.aggregate_categorical'] = True
        search_space['preprocessing.categorical.convert_to_text'] = False
        search_space['preprocessing.numerical.convert_to_text'] = False
    elif strategy == 'fuse_early':
        search_space['model.network.agg_net.agg_type'] = 'attention_token'
        search_space['model.network.aggregate_categorical'] = False
        search_space['preprocessing.categorical.convert_to_text'] = False
        search_space['preprocessing.numerical.convert_to_text'] = False
    elif strategy == 'all_text':
        search_space['model.network.agg_net.agg_type'] = 'concat'
        search_space['model.network.aggregate_categorical'] = False
        search_space['preprocessing.categorical.convert_to_text'] = True
        search_space['preprocessing.numerical.convert_to_text'] = True
    else:
        raise NotImplementedError
    return new_cfg


def apply_backbone(cfg, backbone_name='electra_base'):
    new_cfg = copy.deepcopy(cfg)
    search_space = new_cfg['models']['MultimodalTextModel']['search_space']
    if backbone_name == 'electra_small':
        search_space['model.backbone.name'] = 'google_electra_small'
        search_space['optimization.per_device_batch_size'] = 8
    elif backbone_name == 'electra_base':
        search_space['model.backbone.name'] = 'google_electra_base'
        search_space['optimization.per_device_batch_size'] = 4
    elif backbone_name == 'electra_large':
        search_space['model.backbone.name'] = 'google_electra_large'
        search_space['optimization.per_device_batch_size'] = 2
    elif backbone_name == 'roberta_base':
        search_space['model.backbone.name'] = 'fairseq_roberta_base'
        search_space['model.network.text_net.use_segment_id'] = False
        search_space['optimization.per_device_batch_size'] = 4
    elif backbone_name == 'multi_cased_bert_base':
        search_space['model.backbone.name'] = 'google_multi_cased_bert_base'
        search_space['optimization.per_device_batch_size'] = 4
    else:
        raise NotImplementedError
    return new_cfg


def gen_config_no_hpo(backbone_name, nbest=3, fusion_strategy='fuse_late'):
    cfg = base()
    cfg = apply_backbone(cfg, backbone_name)
    cfg = apply_average_nbest(cfg, nbest=nbest)
    cfg = apply_fusion_strategy(cfg, strategy=fusion_strategy)
    return cfg


for backbone_name in ['electra_small', 'electra_base',
                      'electra_large', 'roberta_base',
                      'multi_cased_bert_base']:
    for fusion_strategy in ['fuse_late']:
        ag_text_presets.register(f'{backbone_name}_{fusion_strategy}',
                                 functools.partial(gen_config_no_hpo,
                                                   backbone_name=backbone_name,
                                                   nbest=3,
                                                   fusion_strategy=fusion_strategy))


for backbone_name in ['electra_base']:
    for fusion_strategy in ['fuse_early', 'all_text']:
        ag_text_presets.register(f'{backbone_name}_{fusion_strategy}',
                                 functools.partial(gen_config_no_hpo,
                                                   backbone_name=backbone_name,
                                                   nbest=3,
                                                   fusion_strategy=fusion_strategy))


@ag_text_presets.register()
def lower_quality_fast_train() -> dict:
    """Configuration that supports fast training and inference.

    By default, we use the late-fusion aggregator with electra-small
    """
    cfg = ag_text_presets.create('electra_small_fuse_late')
    return cfg


@ag_text_presets.register()
def medium_quality_faster_train() -> dict:
    """The medium quality configuration. This is used by default.

    By default, we use the late-fusion aggregator and the electra-base
    """
    cfg = ag_text_presets.create('electra_base_fuse_late')
    return cfg


@ag_text_presets.register()
def best_quality() -> dict:
    """The best quality configuration.

    We will use the ELECTRA-large model with late fusion.

    """
    cfg = ag_text_presets.create('electra_large_fuse_late')
    cfg['models']['MultimodalTextModel']['search_space']['optimization.num_train_epochs'] = 20
    return cfg


@ag_text_presets.register()
def default() -> dict:
    return ag_text_presets.create('medium_quality_faster_train')


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
