import torch.nn as nn


class TabNet(nn.Module):
    def __init__(self, num_class, feature_dim, num_output_layers, device, params):
        """
        Internal torch model that uses TabTransformer as an embedding.
        This is where we are passing through activations and neurons.

        Parameters
        ----------
        num_class (int): Number of classes identified.
        cat_feat_origin_cards (list): List of categorical features
        """
        super().__init__()
        import torch.nn as nn
        from .tab_transformer import TabTransformer
        self.embed = TabTransformer(**params)

        relu = nn.ReLU()
        in_dim = 2 * feature_dim
        lin = nn.Linear(in_dim, in_dim, bias=True)
        lin_out = nn.Linear(in_dim, num_class, bias=True)
        self.fc = [nn.Sequential(*[relu, lin])] * (num_output_layers - 1) + [nn.Sequential(*[relu, lin_out])]

        # Each individual layer inside needs to be put into the GPU.
        # Calling "self.model.cuda()" (TabTransformer:get_model()) will not put a python list into GPU.
        if device.type == "cuda":
            for layer in range(num_output_layers):
                self.fc[layer] = self.fc[layer].cuda()

    def forward(self, data):
        features = self.embed(data)
        out = features.mean(dim=1)
        for layer in range(len(self.fc)):
            out = self.fc[layer](out)
        return out, features


class TabModelBase(nn.Module):
    def __init__(self, n_cont_features, norm_class_name, cat_feat_origin_cards, max_emb_dim,
                 p_dropout, one_hot_embeddings, drop_whole_embeddings):
        super().__init__()
        """
        Base class for all TabTransformer models
        
        Parameters
        ----------
        max_emb_dim (int): Maximum allowable amount of embeddings.
        n_cont_features (int): How many continuous features to concatenate onto the categorical features.
        cat_feat_origin_cards (list): Categorical features to turn into embeddings.
        norm_class: What normalization to use for continuous features.
        p_dropout (float): How much dropout to apply.
        drop_whole_embeddings (bool): If True, dropout pretends the embedding was a missing value. If false, dropout sets embed features to 0
        one_hot_embeddings (bool): If True, one-hot encode variables whose cardinality is < max_emb_dim.
        cat_initializers (dict): Structure to hold the initial embeddings for categorical features.
        """
        self.max_emb_dim = max_emb_dim
        self.n_cont_features = n_cont_features
        self.cat_feat_origin_cards = cat_feat_origin_cards
        self.norm_class = nn.__dict__[norm_class_name]

        self.p_dropout = p_dropout
        self.drop_whole_embeddings = drop_whole_embeddings
        self.one_hot_embeddings = one_hot_embeddings

        self.cat_initializers = nn.ModuleDict()

        from .tab_transformer_encoder import EmbeddingInitializer
        if isinstance(self.cat_feat_origin_cards, list):
            for col_name, card in self.cat_feat_origin_cards:
                self.cat_initializers[col_name] = EmbeddingInitializer(card, max_emb_dim, p_dropout,
                                                                       drop_whole_embeddings=drop_whole_embeddings,
                                                                       one_hot=one_hot_embeddings)
            self.init_feat_dim = sum(i.emb_dim for i in self.cat_initializers.values()) + self.n_cont_features

    def forward(self, input):
        raise NotImplementedError

    def get_norm(self, num_feats):
        return self.norm_class(num_feats)

    def pred_from_output(self, output):
        return output.max(dim=1, keepdim=True)[1]
