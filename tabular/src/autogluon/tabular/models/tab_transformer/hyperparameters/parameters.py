from ....constants import BINARY, MULTICLASS, REGRESSION


def get_fixed_params():
    """ Parameters that currently cannot be searched during HPO
    TODO: HPO NOT CURRENTLY IMPLEMENTED FOR TABTRANSFORMER
    Will need to figure out what (in future PR) is "fixed" and what is searchable. """
    fixed_params = {'batch_size': 512,
                    'n_cont_embeddings': 0,
                    'n_layers': 1,
                    'n_heads': 8,
                    'hidden_dim': 128,
                    'norm_class_name': 'LayerNorm',
                    'tab_readout': 'none',
                    'column_embedding': True,
                    'shared_embedding': False,
                    #'n_shared_embs': 8,
                    'p_dropout': 0.1, # dropout probability, = 0 turns off Dropout.
                    'orig_emb_resid': False,
                    'one_hot_embeddings': False,
                    'drop_whole_embeddings': False,
                    'max_emb_dim': 8,
                    'lr': 1e-3, # Learning rate
                    'weight_decay': 1e-6, # Rate of linear weight decay for learning rate
                    'base_exp_decay': 0.95, # Rate of exponential decay for learning rate
                    'encoders':  {'CATEGORICAL': 'CategoricalOrdinalEnc',
                                  'DATETIME'   : 'DatetimeOrdinalEnc',
                                  'LATLONG'    : 'LatLongQuantileOrdinalEnc',
                                  'SCALAR'     : 'ScalarQuantileOrdinalEnc',
                                  'TEXT'       : 'TextSummaryScalarEnc'},
                    'aug_mask_prob' : 0.4, # What percentage of values to apply augmentation to.
                    'num_augs' : 1, # Number of augmentations to add.
                    'pretext': 'BERT_pretext',
                    'n_cont_features': 8,
                    'fix_attention': False,
                    'freq': 1,
                    'pretrain_freq': 100,
                    'feature_dim': 64,
                    'epochs': 100, # How many epochs to train on with labeled data.
                    'pretrain_epochs': 200, # How many epochs to pretrain on with unlabeled data.
                    'epochs_wo_improve': 25, # How many epochs to continue running without improving on metric.
                    'num_workers': 16, # How many workers to use for torch DataLoader.
                    'max_columns': 1000, # Maximum number of columns TabTransformer will accept as input. This is to combat huge memory requirements/errors.
                    }

    return fixed_params

def get_default_param(problem_type, nunique=None):

    params = get_fixed_params()
    params['problem_type'] = problem_type

    if problem_type==REGRESSION:
        params['n_classes'] = 1
    elif problem_type==BINARY:
        params['n_classes'] = 2
    elif problem_type==MULTICLASS:
        params['n_classes'] = nunique

    return params
