#kwargs for TabTransformer model and training

def get_kwargs(**kw):
    kwargs =   {'batch_size': 512, 
                'tab_kwargs': {'n_cont_embeddings': 0,
                               'n_layers': 1,
                               'n_heads': 8,
                               'hidden_dim': 128,
                               'norm_class_name': 'LayerNorm',
                               'tab_readout': 'none',
                               'column_embedding': True,
                               'shared_embedding': False,
                               #'n_shared_embs': 8, #8, #careful
                               'p_dropout': 0.1,
                               'orig_emb_resid': False,
                               'one_hot_embeddings': False,
                               'drop_whole_embeddings': False,
                               'max_emb_dim': 8},
                'encoders':  {'CATEGORICAL': 'CategoricalOrdinalEnc',
                              'DATETIME'   : 'DatetimeOrdinalEnc',
                              'LATLONG'    : 'LatLongQuantileOrdinalEnc',
                              'SCALAR'     : 'ScalarQuantileOrdinalEnc',
                              'TEXT'       : 'TextSummaryScalarEnc'},
                'pretext':    'BERT_pretext',
                'n_cont_features': 8,
                'fix_attention': True,
                'freq': 1,
                'pretrain_freq': 100,
                'feature_dim': 64,
                'epochs': 20,
                'pretrain_epochs': 200}
      

    #note this overwrites above choices if the keys overlap                      }
    for k,v in kw.items(): 
        kwargs[k]=v
    return kwargs

