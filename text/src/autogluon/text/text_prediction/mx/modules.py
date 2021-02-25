import numpy as np
import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from mxnet.util import use_np
from autogluon_contrib_nlp.utils.config import CfgNode
from autogluon_contrib_nlp.layers import get_activation, get_norm_layer
from autogluon_contrib_nlp.models.transformer import TransformerEncoder
from .. import constants as _C


@use_np
class BasicMLP(HybridBlock):
    def __init__(self, in_units,
                 mid_units,
                 out_units,
                 num_layers=1,
                 normalization='layer_norm',
                 norm_eps=1E-5,
                 dropout=0.1,
                 data_dropout=False,
                 activation='leaky',
                 weight_initializer=None,
                 bias_initializer=None,
                 prefix=None, params=None):
        """
        data -> [dropout] * (0/1) -> [Dense -> Normalization -> ACT] * N -> dropout -> Dense -> out

        Parameters
        ----------
        in_units
        mid_units
        out_units
        num_layers
            Number of intermediate layers
        normalization
        norm_eps
        dropout
        activation
        """
        super().__init__(prefix=prefix, params=params)
        self.in_units = in_units
        self.data_dropout = data_dropout
        if mid_units < 0:
            mid_units = in_units
        with self.name_scope():
            self.proj = nn.HybridSequential()
            with self.proj.name_scope():
                if num_layers > 0 and data_dropout:
                    self.proj.add(nn.Dropout(dropout))
                for i in range(num_layers):
                    self.proj.add(nn.Dense(units=mid_units,
                                           in_units=in_units,
                                           flatten=False,
                                           weight_initializer=weight_initializer,
                                           bias_initializer=bias_initializer,
                                           use_bias=False))
                    self.proj.add(get_norm_layer(normalization,
                                                 axis=-1,
                                                 epsilon=norm_eps,
                                                 in_channels=mid_units))
                    self.proj.add(get_activation(activation))
                    in_units = mid_units
                self.proj.add(nn.Dropout(dropout))
                self.proj.add(nn.Dense(units=out_units,
                                       in_units=in_units,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       flatten=False))

    def hybrid_forward(self, F, x):
        return self.proj(x)


@use_np
class CategoricalFeatureNet(HybridBlock):
    """Embedding of the categories."""
    def __init__(self, num_class, out_units, cfg=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self.cfg = cfg = CategoricalFeatureNet.get_cfg().clone_merge(cfg)
        embed_initializer = mx.init.create(*cfg.initializer.embed)
        weight_initializer = mx.init.create(*cfg.initializer.weight)
        bias_initializer = mx.init.create(*cfg.initializer.bias)
        with self.name_scope():
            self.embedding = nn.Embedding(input_dim=num_class,
                                          output_dim=cfg.emb_units,
                                          weight_initializer=embed_initializer)
            self.proj = BasicMLP(in_units=cfg.emb_units,
                                 mid_units=cfg.mid_units,
                                 out_units=out_units,
                                 num_layers=cfg.num_layers,
                                 normalization=cfg.normalization,
                                 norm_eps=cfg.norm_eps,
                                 data_dropout=cfg.data_dropout,
                                 dropout=cfg.dropout,
                                 activation=cfg.activation,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer)

    @staticmethod
    def get_cfg(key=None):
        if key is None:
            cfg = CfgNode()
            cfg.emb_units = 32
            cfg.mid_units = 64
            cfg.num_layers = 1
            cfg.data_dropout = False
            cfg.dropout = 0.1
            cfg.activation = 'leaky'
            cfg.normalization = 'layer_norm'
            cfg.norm_eps = 1e-5
            cfg.initializer = CfgNode()
            cfg.initializer.embed = ['xavier', 'gaussian', 'in', 1.0]
            cfg.initializer.weight = ['xavier', 'uniform', 'avg', 3.0]
            cfg.initializer.bias = ['zeros']
            return cfg
        else:
            raise NotImplementedError

    def hybrid_forward(self, F, feature):
        embed = self.embedding(feature)
        return self.proj(embed)


@use_np
class NumericalFeatureNet(HybridBlock):
    def __init__(self, input_shape, out_units, cfg=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self.cfg = cfg = NumericalFeatureNet.get_cfg().clone_merge(cfg)
        self.input_shape = input_shape
        self.need_first_reshape = isinstance(input_shape, (list, tuple)) and len(input_shape) != 1
        self.in_units = int(np.prod(input_shape))
        weight_initializer = mx.init.create(*cfg.initializer.weight)
        bias_initializer = mx.init.create(*cfg.initializer.bias)
        with self.name_scope():
            if self.cfg.input_centering:
                self.data_bn = nn.BatchNorm(in_channels=self.in_units)
            if self.cfg.gated_activation:
                self.gate_proj = BasicMLP(in_units=self.in_units,
                                          mid_units=cfg.mid_units,
                                          out_units=out_units,
                                          num_layers=cfg.num_layers,
                                          normalization=cfg.normalization,
                                          norm_eps=cfg.norm_eps,
                                          data_dropout=cfg.data_dropout,
                                          dropout=cfg.dropout,
                                          activation=cfg.activation,
                                          weight_initializer=weight_initializer,
                                          bias_initializer=bias_initializer)
            else:
                self.gate_proj = None
            self.proj = BasicMLP(in_units=self.in_units,
                                 mid_units=cfg.mid_units,
                                 out_units=out_units,
                                 num_layers=cfg.num_layers,
                                 normalization=cfg.normalization,
                                 norm_eps=cfg.norm_eps,
                                 data_dropout=cfg.data_dropout,
                                 dropout=cfg.dropout,
                                 activation=cfg.activation,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer)

    @staticmethod
    def get_cfg(key=None):
        if key is None:
            cfg = CfgNode()
            cfg.input_centering = False
            cfg.gated_activation = False
            cfg.mid_units = 128
            cfg.num_layers = 1
            cfg.data_dropout = False
            cfg.dropout = 0.1
            cfg.activation = 'leaky'
            cfg.normalization = 'layer_norm'
            cfg.norm_eps = 1e-5
            cfg.initializer = CfgNode()
            cfg.initializer.weight = ['xavier', 'uniform', 'avg', 3.0]
            cfg.initializer.bias = ['zeros']
        else:
            raise NotImplementedError
        return cfg

    def hybrid_forward(self, F, features):
        if self.need_first_reshape:
            features = F.np.reshape(features, (-1, self.in_units))
        if self.cfg.input_centering:
            features = self.data_bn(features)
        if self.gate_proj is not None:
            return F.npx.sigmoid(self.gate_proj(features)) * self.proj(features)
        else:
            return self.proj(features)


@use_np
class FeatureAggregator(HybridBlock):
    def __init__(self, num_fields, out_shape, in_units,
                 cfg=None, get_embedding=False, prefix=None, params=None):
        """

        Parameters
        ----------
        num_fields
            The number of fields of the
        out_shape
        in_units
            The number of input units
        cfg
            The configuration
        get_embedding
            Whether to get the embedding
        prefix
            The prefix
        params
            The parameters
        """
        super().__init__(prefix=prefix, params=params)
        if cfg is None:
            cfg = FeatureAggregator.get_cfg()
        self.cfg = cfg = FeatureAggregator.get_cfg().clone_merge(cfg)
        self.num_fields = num_fields
        if isinstance(out_shape, list):
            out_shape = tuple(out_shape)
        self.out_shape = out_shape
        self.in_units = in_units
        self.get_embedding = get_embedding
        weight_initializer = mx.init.create(*cfg.initializer.weight)
        bias_initializer = mx.init.create(*cfg.initializer.bias)
        out_units = int(np.prod(out_shape))
        with self.name_scope():
            if num_fields > 1:
                if cfg.agg_type == 'attention' or cfg.agg_type == 'attention_token':
                    if cfg.attention_net.hidden_size < 0:
                        hidden_size = 4 * cfg.attention_net.units
                    if cfg.attention_net.units != self.in_units:
                        self.attention_net_pre_proj = nn.Dense(units=cfg.attention_net.units,
                                                               in_units=in_units,
                                                               use_bias=False,
                                                               weight_initializer=weight_initializer,
                                                               bias_initializer=bias_initializer,
                                                               flatten=False,
                                                               prefix='attention_net_pre_proj_')
                    else:
                        self.attention_net_pre_proj = None
                    self.attention_agg_ln = nn.LayerNorm(in_channels=in_units,
                                                         epsilon=self.cfg.norm_eps)
                    self.attention_agg_dropout = nn.Dropout(self.cfg.dropout)
                    self.attention_transformer_enc = TransformerEncoder(
                        num_layers=cfg.attention_net.num_layers,
                        num_heads=cfg.attention_net.num_heads,
                        units=cfg.attention_net.units,
                        hidden_size=hidden_size,
                        dropout=cfg.dropout,
                        activation=cfg.attention_net.activation,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer)
                # Construct out proj
                if cfg.agg_type == 'mean' or cfg.agg_type == 'max':
                    in_units = in_units
                elif cfg.agg_type == 'concat':
                    in_units = in_units * num_fields
                elif cfg.agg_type == 'attention' or cfg.agg_type == 'attention_token':
                    in_units = cfg.attention_net.units
                else:
                    raise NotImplementedError
            mid_units = in_units if cfg.mid_units < 0 else cfg.mid_units
            self.out_proj = BasicMLP(in_units=in_units,
                                     mid_units=mid_units,
                                     out_units=out_units,
                                     num_layers=cfg.out_proj_num_layers,
                                     data_dropout=cfg.data_dropout,
                                     normalization=cfg.normalization,
                                     norm_eps=cfg.norm_eps,
                                     dropout=cfg.dropout,
                                     activation=cfg.activation,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer)

    @staticmethod
    def get_cfg(key=None):
        if key is None:
            cfg = CfgNode()
            cfg.agg_type = 'concat'

            # Attention Aggregator
            cfg.attention_net = CfgNode()
            cfg.attention_net.num_layers = 6
            cfg.attention_net.units = 64
            cfg.attention_net.num_heads = 4
            cfg.attention_net.hidden_size = -1  # Size of the FFN network used in attention
            cfg.attention_net.activation = 'gelu'   # Activation of the attention

            # Other parameters
            cfg.mid_units = 128
            cfg.feature_proj_num_layers = -1
            cfg.out_proj_num_layers = 1
            cfg.data_dropout = False
            cfg.dropout = 0.1
            cfg.activation = 'leaky'
            cfg.normalization = 'layer_norm'
            cfg.norm_eps = 1e-5
            cfg.initializer = CfgNode()
            cfg.initializer.weight = ['xavier', 'uniform', 'avg', 3.0]
            cfg.initializer.bias = ['zeros']
        else:
            raise NotImplementedError
        return cfg

    def hybrid_forward(self, F, features, valid_length=None):
        """

        Parameters
        ----------
        features
            If it is the
            List of projection features. All elements must have the same shape.
        valid_length
            The valid_length of the text column. If it is given it will have shape (B,)

        Returns
        -------
        scores
            Shape (batch_size,) + out_shape
        """
        if len(features) == 1:
            agg_features = features[0]
        else:
            if self.cfg.agg_type == 'attention_token':
                # Features[0] will have shape (B, T, C)
                other_features = F.np.stack(features[1:], axis=1)
                agg_features = F.np.concatenate([other_features, features[0]], axis=1)
                agg_features = self.attention_agg_ln(agg_features)
                agg_features = self.attention_agg_dropout(agg_features)
                if self.attention_net_pre_proj is not None:
                    agg_features = self.attention_net_pre_proj(agg_features)
                agg_features = self.attention_transformer_enc(agg_features,
                                                              valid_length + len(features) - 1)
                agg_features = agg_features[:, len(features) - 1, :]
            else:
                agg_features = F.np.stack(features, axis=1)
                if self.cfg.agg_type == 'mean':
                    agg_features = F.np.mean(agg_features, axis=1)
                elif self.cfg.agg_type == 'max':
                    agg_features = F.np.max(agg_features, axis=1)
                elif self.cfg.agg_type == 'concat':
                    agg_features = F.npx.reshape(agg_features, (-2, -1))
                elif self.cfg.agg_type == 'attention':
                    if self.attention_net_pre_proj is not None:
                        agg_features = self.attention_net_pre_proj(agg_features)
                    agg_features = self.attention_transformer_enc(agg_features, None)
                    agg_features = F.np.mean(agg_features, axis=1)
                else:
                    raise NotImplementedError
        scores = self.out_proj(agg_features)
        if len(self.out_shape) != 1:
            scores = F.np.reshape(scores, (-1,) + self.out_shape)
        if self.get_embedding:
            return scores, agg_features
        else:
            return scores


@use_np
class MultiModalWithPretrainedTextNN(HybridBlock):
    """The basic model for classification + regression of multimodal tabular data
    with text, numerical, and categorical columns.

    It uses pretrained model, e.g, ELECTRA, BERT, ALBERT, RoBERTa, etc. as the backbone for
    handling text data.

    Here, we use the backbone network to extract the contextual embeddings and use
    another dense layer to map the contextual embeddings to the class scores.

    Input:

    TextField + EntityField --> TextNet -------> TextFeature
        ...
    CategoricalField --> CategoricalNet --> CategoricalFeature  ==> AggregateNet --> Dense --> logits/scores
        ...
    NumericalField ----> NumericalNet ----> NumericalFeature


    We support three aggregators:
    - mean
        Take the average of the input features
    - concat
        Concatenate the input features
    - max
        Take the maximum of the input features
    - attention
        We use a stack of transformer-encoder layer to aggregate the information.
    - attention_token
        We use a stack of transformer-encoder layers to aggregate the information.
        Instead of using one embedding to describe the text data. We keep the original embeddings
        and fuse it jointly with the other embeddings.
    -
    """
    def __init__(self, text_backbone,
                 num_text_features,
                 num_categorical_features,
                 num_numerical_features,
                 numerical_input_units,
                 num_categories,
                 out_shape,
                 cfg=None,
                 get_embedding=False,
                 prefix=None,
                 params=None):
        """

        Parameters
        ----------
        text_backbone
            Backbone network for handling the text data
        num_text_features
            Number of text features.
            Each text feature will have (text_token_ids, valid_length)
        num_categorical_features
            Number of categorical features
        num_numerical_features
            Number of numerical features
        numerical_input_units
            The number of units for each numerical column
        num_categories
            The number of categories for each categorical column.
        out_shape
            Shape of the output
        cfg
            The configuration of the network
        get_embedding
            Whether to output the aggregated intermediate embedding from the network
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        self.cfg = cfg = MultiModalWithPretrainedTextNN.get_cfg().clone_merge(cfg)
        assert self.cfg.text_net.pool_type == 'cls'
        base_feature_units = self.cfg.base_feature_units
        if not isinstance(out_shape, (list, tuple)):
            out_shape = (out_shape,)
        self.out_shape = out_shape
        if base_feature_units == -1:
            base_feature_units = text_backbone.units
        self.get_embedding = get_embedding
        self.num_text_features = num_text_features
        self.num_categorical_features = num_categorical_features
        self.num_numerical_features = num_numerical_features
        if numerical_input_units is None:
            numerical_input_units = []
        elif not isinstance(numerical_input_units, (list, tuple)):
            numerical_input_units = [numerical_input_units] * self.num_numerical_features
        self.numerical_input_units = numerical_input_units
        self.num_categories = num_categories
        if self.num_categorical_features > 0:
            assert len(self.num_categories) == self.num_categorical_features
        weight_initializer = mx.init.create(*cfg.initializer.weight)
        bias_initializer = mx.init.create(*cfg.initializer.bias)
        self.agg_type = cfg.agg_net.agg_type
        if self.agg_type == 'attention_token':
            assert self.num_text_features == 1, \
                'Only supports a single text input if use token-level attention'
        with self.name_scope():
            self.text_backbone = text_backbone
            if base_feature_units != text_backbone.units:
                self.text_proj = nn.HybridSequential()
                for i in range(self.num_text_features):
                    with self.text_proj.name_scope():
                        self.text_proj.add(nn.Dense(in_units=text_backbone.units,
                                                    units=base_feature_units,
                                                    use_bias=False,
                                                    weight_initializer=weight_initializer,
                                                    bias_initializer=bias_initializer,
                                                    flatten=False))
            else:
                self.text_proj = None
            if self.num_categorical_features > 0:
                self.categorical_networks = nn.HybridSequential()
                for i in range(self.num_categorical_features):
                    with self.categorical_networks.name_scope():
                        self.categorical_networks.add(
                            CategoricalFeatureNet(num_class=self.num_categories[i],
                                                  out_units=base_feature_units,
                                                  cfg=cfg.categorical_net))
            else:
                self.categorical_networks = None
            if self.cfg.aggregate_categorical and self.num_categorical_features > 1:
                # Use another dense layer to aggregate the categorical features
                self.categorical_agg = BasicMLP(
                    in_units=base_feature_units * self.num_categorical_features,
                    mid_units=cfg.categorical_agg.mid_units,
                    out_units=base_feature_units,
                    activation=cfg.categorical_agg.activation,
                    dropout=cfg.categorical_agg.dropout,
                    num_layers=cfg.categorical_agg.num_layers,
                    weight_initializer=weight_initializer,
                    bias_initializer=bias_initializer
                )
                if self.cfg.categorical_agg.gated_activation:
                    self.categorical_agg_gate = BasicMLP(
                        in_units=base_feature_units * self.num_categorical_features,
                        mid_units=cfg.categorical_agg.mid_units,
                        out_units=base_feature_units,
                        activation=cfg.categorical_agg.activation,
                        dropout=cfg.categorical_agg.dropout,
                        num_layers=cfg.categorical_agg.num_layers,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer
                    )
                else:
                    self.categorical_agg_gate = None
            else:
                self.categorical_agg = None
                self.categorical_agg_gate = None

            if self.num_numerical_features > 0:
                self.numerical_networks = nn.HybridSequential()
                for i in range(self.num_numerical_features):
                    with self.numerical_networks.name_scope():
                        self.numerical_networks.add(
                            NumericalFeatureNet(input_shape=self.numerical_input_units[i],
                                                out_units=base_feature_units,
                                                cfg=cfg.numerical_net))
            else:
                self.numerical_networks = None
            self.agg_layer = FeatureAggregator(num_fields=self.num_fields,
                                               out_shape=out_shape,
                                               in_units=base_feature_units,
                                               cfg=cfg.agg_net,
                                               get_embedding=get_embedding)

    @staticmethod
    def get_cfg(key=None):
        if key is None:
            cfg = CfgNode()
            cfg.base_feature_units = -1  # -1 means not given and we will use the units of BERT
            cfg.text_net = CfgNode()
            cfg.text_net.use_segment_id = True
            cfg.text_net.pool_type = 'cls'
            cfg.aggregate_categorical = True  # Whether to use one network to aggregate the categorical columns.
            cfg.categorical_agg = CfgNode()
            cfg.categorical_agg.activation = 'leaky'
            cfg.categorical_agg.mid_units = 128
            cfg.categorical_agg.num_layers = 1
            cfg.categorical_agg.dropout = 0.1
            cfg.categorical_agg.gated_activation = False
            cfg.agg_net = FeatureAggregator.get_cfg()
            cfg.categorical_net = CategoricalFeatureNet.get_cfg()
            cfg.numerical_net = NumericalFeatureNet.get_cfg()
            cfg.initializer = CfgNode()
            cfg.initializer.weight = ['xavier', 'uniform', 'avg', 3.0]
            cfg.initializer.bias = ['zeros']
            return cfg
        else:
            raise NotImplementedError

    @property
    def num_fields(self):
        if self.cfg.aggregate_categorical and self.num_categorical_features > 1:
            return self.num_text_features + 1 + self.num_numerical_features
        else:
            return self.num_text_features + self.num_categorical_features + self.num_numerical_features

    def initialize_with_pretrained_backbone(self, backbone_params_path, ctx=None):
        self.text_backbone.load_parameters(backbone_params_path, ctx=ctx)
        self.agg_layer.initialize(ctx=ctx)
        if self.text_proj is not None:
            self.text_proj.initialize(ctx=ctx)
        if self.categorical_networks is not None:
            self.categorical_networks.initialize(ctx=ctx)
        if self.numerical_networks is not None:
            self.numerical_networks.initialize(ctx=ctx)
        if self.categorical_agg is not None:
            self.categorical_agg.initialize(ctx=ctx)
        if self.categorical_agg_gate is not None:
            self.categorical_agg_gate.initialize(ctx=ctx)

    def hybrid_forward(self, F, features):
        """

        Parameters
        ----------
        features
            A list of field data
            It will contain
            - text_features
                ...
            - categorical_features
                ...
            - numerical_features
                ...

        Returns
        -------
        logits_or_scores
            Shape (batch_size,) + out_shape
        """
        field_features = []
        ptr = 0
        text_valid_length = None
        for i in range(self.num_text_features):
            batch_token_ids, batch_valid_length, batch_segment_ids = features[i]
            if self.cfg.text_net.use_segment_id:
                contextual_embedding, _ = self.text_backbone(batch_token_ids,
                                                             batch_segment_ids,
                                                             batch_valid_length)
            else:
                _, all_hidden_states = self.text_backbone(batch_token_ids, batch_valid_length)
                contextual_embedding = all_hidden_states[-1]
            if self.agg_type == 'attention_token' and self.num_fields > 1:
                if self.text_proj is not None:
                    contextual_embedding = self.text_proj[i](contextual_embedding)
                text_valid_length = batch_valid_length
                field_features.append(contextual_embedding)
            else:
                pooled_output = contextual_embedding[:, 0, :]
                if self.text_proj is not None:
                    pooled_output = self.text_proj[i](pooled_output)
                field_features.append(pooled_output)
        ptr += self.num_text_features
        all_cat_features = []
        for i in range(ptr, ptr + self.num_categorical_features):
            cat_features = self.categorical_networks[i - ptr](features[i])
            if self.categorical_agg is not None:
                all_cat_features.append(cat_features)
            else:
                field_features.append(cat_features)
        if self.categorical_agg is not None:
            all_cat_features = F.np.concatenate(all_cat_features, axis=-1)
            if self.cfg.categorical_agg.gated_activation:
                field_features.append(
                    F.npx.sigmoid(self.categorical_agg_gate(all_cat_features))
                    * self.categorical_agg(all_cat_features))
            else:
                field_features.append(self.categorical_agg(all_cat_features))
        ptr += self.num_categorical_features
        for i in range(ptr, ptr + self.num_numerical_features):
            numerical_features = self.numerical_networks[i - ptr](features[i])
            field_features.append(numerical_features)
        if self.agg_type == 'attention_token':
            return self.agg_layer(field_features, text_valid_length)
        else:
            return self.agg_layer(field_features)
