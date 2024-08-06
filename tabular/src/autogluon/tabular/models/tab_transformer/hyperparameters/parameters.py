def get_fixed_params():
    """Parameters that currently cannot be searched during HPO"""
    fixed_params = {
        "batch_size": 512,  # The size of example chunks to predict on.
        "n_cont_embeddings": 0,  # How many continuous feature embeddings to use.
        "norm_class_name": "LayerNorm",  # What kind of normalization to use on continuous features.
        "column_embedding": True,  # If True, 1/(n_shared_embs)th of every embedding will be reserved for a learned parameter that's common to all embeddings.
        #'shared_embedding': False,
        #'n_shared_embs': 8,
        "orig_emb_resid": False,  # If True, concatenate the original embeddings on top of the feature embeddings in the Transformer layers.
        "one_hot_embeddings": False,  # If True, one-hot encode variables whose cardinality is < max_emb_dim.
        "drop_whole_embeddings": False,  # If True, dropout pretends the embedding was a missing value. If false, dropout sets embed features to 0
        "max_emb_dim": 8,  # Maximum allowable amount of embeddings.
        "base_exp_decay": 0.95,  # Rate of exponential decay for learning rate, used during finetuning.
        "encoders": {
            "CATEGORICAL": "CategoricalOrdinalEnc",  # How to "encode"(vectorize) each column type.
            "DATETIME": "DatetimeOrdinalEnc",
            "LATLONG": "LatLongQuantileOrdinalEnc",
            "SCALAR": "ScalarQuantileOrdinalEnc",
            "TEXT": "TextSummaryScalarEnc",
        },
        "aug_mask_prob": 0.4,  # What percentage of values to apply augmentation to.
        "num_augs": 0,  # Number of augmentations to add.
        "pretext": "BERTPretext",  # What pretext to use when performing pretraining/semi-supervised learning.
        "n_cont_features": 8,  # How many continuous features to concatenate onto the categorical features
        "fix_attention": False,  # If True, use the categorical embeddings in the transformer architecture.
        "epochs": 200,  # How many epochs to train on with labeled data.
        "pretrain_epochs": 200,  # How many epochs to pretrain on with unlabeled data.
        "epochs_wo_improve": 30,  # How many epochs to continue running without improving on metric. aka "Early Stopping Patience"
        "num_workers": 16,  # How many workers to use for torch DataLoader.
        "max_columns": 500,  # Maximum number of columns TabTransformer will accept as input. This is to combat huge memory requirements/errors.
        "tab_readout": "none",  # What sort of readout from the transformer. Options: ['readout_emb', 'mean', 'concat_pool', 'concat_pool_all', 'concat_pool_add', 'all_feat_embs', 'mean_feat_embs', 'none']
    }

    return fixed_params


def get_hyper_params():
    """Parameters that currently can be tuned during HPO"""
    hyper_params = {
        "lr": 3.6e-3,  # Learning rate
        # Options: Real(5e-5, 5e-3)
        "weight_decay": 1e-6,  # Rate of linear weight decay for learning rate
        # Options: Real(1e-6, 5e-2)
        "p_dropout": 0,  # dropout probability, 0 turns off Dropout.
        # Options: Categorical(0, 0.1, 0.2, 0.3, 0.4, 0.5)
        "n_heads": 4,  # Number of attention heads
        # Options: Categorical(2, 4, 8)
        "hidden_dim": 128,  # hidden dimension size
        # Options: Categorical(32, 64, 128, 256)
        "n_layers": 2,  # Number of Tab Transformer encoder layers,
        # Options: Categorical(1, 2, 3, 4, 5)
        "feature_dim": 64,  # Size of fully connected layer in TabNet.
        # Options: Int(8, 128)
        "num_output_layers": 1,  # How many fully-connected layers on top of transformer to produce predictions. Minimum 1 layer.
        # Options: Categorical(1, 2, 3)
    }

    return hyper_params


def get_default_param():
    params = get_fixed_params()
    params.update(get_hyper_params())

    return params
