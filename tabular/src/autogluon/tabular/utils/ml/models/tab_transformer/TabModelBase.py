from .....utils.try_import import try_import_torch
from .TabTransformerEncoder import EmbeddingInitializerClass

class TabModelBaseClass:
    try_import_torch()
    import torch.nn as nn

    class TabModelBase(nn.Module):
        """
        Base class for all tabular models
        """

        def __init__(self, n_cont_features, norm_class_name, cat_feat_origin_cards, max_emb_dim,
                     p_dropout,
                     one_hot_embeddings, drop_whole_embeddings, **kwargs):
            super().__init__()
            try_import_torch()
            import torch.nn as nn
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
            if isinstance(self.cat_feat_origin_cards, list):
                for col_name, card in self.cat_feat_origin_cards:
                    self.cat_initializers[col_name] = EmbeddingInitializerClass.EmbeddingInitializer(card, max_emb_dim, p_dropout,
                                                                           drop_whole_embeddings=drop_whole_embeddings,
                                                                           one_hot=one_hot_embeddings)
                self.init_feat_dim = sum(i.emb_dim for i in self.cat_initializers.values()) + self.n_cont_features


        def forward(self, input):
            raise NotImplementedError

        def get_norm(self, num_feats):
            return self.norm_class(num_feats)


        def pred_from_output(self, output):
            return output.max(dim=1, keepdim=True)[1]
