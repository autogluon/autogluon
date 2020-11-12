# Code in this script based on work by Milan Cvitkovic,
# Xin Huang, Ashish Khetan and Zohar Karnin.

import copy

import torch
import torch.nn as nn

from .tab_model_base import TabModelBase


class TabTransformer(TabModelBase):
    """
    Transformer model for tabular data, can also be used for semi-supervised learning.
    This is the internal transformer model embedding that will have further fully connected layers (TabNet) to
    actually produce predictions.
    """

    def __init__(self, n_cont_embeddings, n_layers, n_heads, hidden_dim, tab_readout, column_embedding, orig_emb_resid,
                 fix_attention, n_shared_embs=8, shared_embedding_added=False, **kwargs):
        super().__init__(n_cont_features=kwargs['n_cont_features'], norm_class_name=kwargs['norm_class_name'],
                         cat_feat_origin_cards=kwargs['cat_feat_origin_cards'], max_emb_dim=kwargs['max_emb_dim'],
                         p_dropout=kwargs['p_dropout'], one_hot_embeddings=kwargs['one_hot_embeddings'],
                         drop_whole_embeddings=kwargs['drop_whole_embeddings'])

        from .modified_transformer import TransformerEncoderLayerModified

        self.n_cont_embeddings = n_cont_embeddings
        self.hidden_dim = hidden_dim
        self.readout = tab_readout
        self.orig_emb_resid = orig_emb_resid

        # Overwriting some TabModelBase options

        self.cat_initializers = nn.ModuleDict()
        if isinstance(self.cat_feat_origin_cards, list):
            self.n_embeddings = len(self.cat_feat_origin_cards) + (n_cont_embeddings if self.n_cont_features else 0)
        else:
            self.n_embeddings = None

        self.cat_initializers = nn.ModuleDict()

        from .tab_transformer_encoder import EmbeddingInitializer
        for col_name, card in self.cat_feat_origin_cards:
            self.cat_initializers[col_name] = EmbeddingInitializer(
                num_embeddings=card,
                max_emb_dim=self.max_emb_dim,
                p_dropout=self.p_dropout,
                minimize_emb_dim=False,
                drop_whole_embeddings=self.drop_whole_embeddings,
                one_hot=False,
                out_dim=self.hidden_dim,
                shared_embedding=column_embedding,
                n_shared_embs=n_shared_embs,
                shared_embedding_added=shared_embedding_added,
            )
        if self.n_cont_features:
            self.cont_norm = self.get_norm(self.n_cont_features)
            self.cont_initializer = nn.Linear(self.n_cont_features, hidden_dim * n_cont_embeddings)
            self.cont_init_norm = self.get_norm(hidden_dim * n_cont_embeddings)

        if self.readout == 'readout_emb':
            self.readout_emb = nn.Parameter(
                torch.zeros(1, hidden_dim).uniform_(-1, 1))  # We do the readout from a learned embedding
            self.n_embeddings += 1

        if fix_attention is True:
            self.n_cat_embeddings = len(self.cat_feat_origin_cards)
            self.tfmr_layers = nn.ModuleList([TransformerEncoderLayerModified(d_model=hidden_dim,
                                                                               n_cat_embeddings=self.n_cat_embeddings,
                                                                               nhead=n_heads,
                                                                               dim_feedforward=4 * hidden_dim,
                                                                               dropout=self.p_dropout,
                                                                               activation='gelu') for _ in
                                              range(n_layers)])
        else:
            self.tfmr_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                                         nhead=n_heads,
                                                                         dim_feedforward=4 * hidden_dim,
                                                                         dropout=self.p_dropout,
                                                                         activation='gelu') for _ in
                                              range(n_layers)])

    def init_input(self, input):
        feats = [init(input[:, i]) for i, init in enumerate(self.cat_initializers.values())]

        if self.readout == 'readout_emb':
            readout_emb = self.readout_emb.expand_as(feats[0])
            feat_embs = torch.stack([readout_emb] + feats,
                                    dim=0)  # (n_feat_embeddings + 1) x batch x hidden_dim
        else:
            feat_embs = torch.stack(feats, dim=0)  # n_feat_embeddings x batch x hidden_dim
        return feat_embs

    def run_tfmr(self, feat_embs):
        orig_feat_embs = feat_embs
        all_feat_embs = [feat_embs]
        for layer in self.tfmr_layers:
            feat_embs = layer(feat_embs)
            all_feat_embs.append(feat_embs)
            if self.orig_emb_resid:
                feat_embs = feat_embs + orig_feat_embs

        if self.readout == 'readout_emb':
            out = self.fc_out(feat_embs[0])
        elif self.readout == 'mean':
            out = torch.mean(feat_embs, dim=0)
            out = self.fc_out(out)
        elif self.readout == 'concat_pool':
            all_feat_embs = torch.cat(all_feat_embs, dim=0)

            max = all_feat_embs.max(dim=0).values
            mean = all_feat_embs.mean(dim=0)
            last_layer = feat_embs.transpose(0, 1).reshape(feat_embs.shape[1], -1)
            out = torch.cat((last_layer, max, mean), dim=1)
            out = self.fc_out(out)
        elif self.readout == 'concat_pool_all':
            feat_embs_all_layers = []
            for each_feat_embs in [all_feat_embs[0], all_feat_embs[-1]]:
                feat_embs_all_layers.append(each_feat_embs.transpose(0, 1).reshape(each_feat_embs.shape[1], -1))

            all_feat_embs = torch.cat(all_feat_embs, dim=0)
            max = all_feat_embs.max(dim=0).values
            mean = all_feat_embs.mean(dim=0)

            feat_embs_all_layers.append(max)
            feat_embs_all_layers.append(mean)
            out = torch.cat(feat_embs_all_layers, dim=1)
            out = self.fc_out(out)
        elif self.readout == 'concat_pool_add':
            orig_feat_embs_cp = copy.deepcopy(orig_feat_embs.detach())
            # ce_dim = orig_feat_embs_cp.shape[-1]//8
            # orig_feat_embs_cp[:, :, ce_dim:] = 0

            last_layer = feat_embs.transpose(0, 1).reshape(feat_embs.shape[1], -1)
            last_layer += orig_feat_embs_cp.transpose(0, 1).reshape(orig_feat_embs_cp.shape[1], -1)

            all_feat_embs = torch.cat(all_feat_embs, dim=0)
            max = all_feat_embs.max(dim=0).values
            mean = all_feat_embs.mean(dim=0)

            out = torch.cat([last_layer, max, mean], dim=1)

        elif self.readout == 'all_feat_embs':
            out = feat_embs

        elif self.readout == 'mean_feat_embs':
            out = feat_embs.mean(dim=0)

        elif self.readout == 'none':
            out = feat_embs.transpose(1, 0)

        return out

    def forward(self, input):
        """
        Returns logits for output classes
        """
        feat_embs = self.init_input(input)
        out = self.run_tfmr(feat_embs)
        return out
