import torch.nn as nn


class TabNet(nn.Module):
    def __init__(self, num_class, params, cat_feat_origin_cards):
        super(TabNet, self).__init__()
        import torch.nn as nn
        from .tab_transformer import TabTransformer
        self.params = params
        self.params['cat_feat_origin_cards'] = cat_feat_origin_cards
        self.embed = TabTransformer(**self.params)

        relu, lin = nn.ReLU(), nn.Linear(2 * self.params['feature_dim'], num_class, bias=True)
        self.fc = nn.Sequential(*[relu, lin])

    def forward(self, data):
        features = self.embed(data)
        out = features.mean(dim=1)
        out = self.fc(out)
        return out, features


class TabModelBase(nn.Module):
    """
    Base class for all tabular models
    """

    def __init__(self, n_cont_features, norm_class_name, cat_feat_origin_cards, max_emb_dim,
                 p_dropout, one_hot_embeddings, drop_whole_embeddings, **kwargs):
        super().__init__()
        self.kwargs = kwargs

        self.act_on_output = True
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
