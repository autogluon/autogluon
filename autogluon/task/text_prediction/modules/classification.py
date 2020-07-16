import numpy as np
import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from mxnet.util import use_np
from .. import constants as _C
from ...utils.config import CfgNode
from ...layers import get_activation, get_norm_layer


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
                                                 in_channels=in_units))
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
    def __init__(self, num_class, out_units, cfg=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        if cfg is None:
            cfg = CategoricalFeatureNet.get_cfg()
        else:
            cfg = CategoricalFeatureNet.get_cfg().clone_merge(cfg)
        self.cfg = cfg
        embed_initializer = mx.init.create(*cfg.INITIALIZER.embed)
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
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
            cfg.INITIALIZER = CfgNode()
            cfg.INITIALIZER.embed = ['xavier', 'gaussian', 'in', 1.0]
            cfg.INITIALIZER.weight = ['xavier', 'uniform', 'avg', 3.0]
            cfg.INITIALIZER.bias = ['zeros']
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
        if cfg is None:
            cfg = NumericalFeatureNet.get_cfg()
        self.input_shape = input_shape
        self.in_units = int(np.prod(input_shape))
        self.cfg = NumericalFeatureNet.get_cfg().clone_merge(cfg)
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
        with self.name_scope():
            if self.cfg.input_centering:
                self.data_bn = nn.BatchNorm(in_channels=self.in_units)
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
            cfg.mid_units = 128
            cfg.num_layers = 1
            cfg.data_dropout = False
            cfg.dropout = 0.1
            cfg.activation = 'leaky'
            cfg.normalization = 'layer_norm'
            cfg.norm_eps = 1e-5
            cfg.INITIALIZER = CfgNode()
            cfg.INITIALIZER.weight = ['xavier', 'uniform', 'avg', 3.0]
            cfg.INITIALIZER.bias = ['zeros']
        else:
            raise NotImplementedError
        return cfg

    def hybrid_forward(self, F, features):
        feature = F.np.reshap(features, (-1, self.in_units))
        if self.cfg.input_centering:
            feature = self.data_bn(feature)
        return self.proj(feature)


@use_np
class FeatureAggregator(HybridBlock):
    def __init__(self, num_fields, out_shape, in_units,
                 cfg=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        if cfg is None:
            cfg = FeatureAggregator.get_cfg()
        self.cfg = FeatureAggregator.get_cfg().clone_merge(cfg)
        self.num_fields = num_fields
        self.out_shape = out_shape
        self.in_units = in_units
        weight_initializer = mx.init.create(*self.cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*self.cfg.INITIALIZER.bias)
        out_units = int(np.prod(out_shape))
        with self.name_scope():
            if self.cfg.agg_type == 'mean':
                in_units = in_units
            elif self.cfg.agg_type == 'concat':
                in_units = in_units * num_fields
            else:
                raise NotImplementedError
            mid_units = in_units if cfg.mid_units < 0 else cfg.mid_units
            self.proj = BasicMLP(in_units=in_units,
                                 mid_units=mid_units,
                                 out_units=out_units,
                                 num_layers=cfg.num_layers,
                                 normalization=cfg.normalization,
                                 norm_eps=cfg.norm_eps,
                                 dropout=cfg.dropout,
                                 data_dropout=cfg.data_dropout,
                                 activation=cfg.activation,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer)

    @staticmethod
    def get_cfg(key=None):
        if key is None:
            cfg = CfgNode()
            cfg.agg_type = 'concat'
            cfg.mid_units = -1
            cfg.num_layers = 0
            cfg.data_dropout = False
            cfg.dropout = 0.1
            cfg.activation = 'tanh'
            cfg.normalization = 'layer_norm'
            cfg.norm_eps = 1e-5
            cfg.INITIALIZER = CfgNode()
            cfg.INITIALIZER.weight = ['xavier', 'uniform', 'avg', 3.0]
            cfg.INITIALIZER.bias = ['zeros']
        else:
            raise NotImplementedError
        return cfg

    def hybrid_forward(self, F, field_proj_features):
        """

        Parameters
        ----------
        field_proj_features
            List of projection features. All elements must have the same shape.

        Returns
        -------
        scores
            Shape (batch_size,) + out_shape
        """
        if len(field_proj_features) == 0:
            agg_features = field_proj_features[0]
        else:
            if self.cfg.agg_type == 'mean':
                agg_features = F.np.stack(field_proj_features)
                agg_features = F.np.mean(agg_features, axis=0)
            elif self.cfg.agg_type == 'concat':
                agg_features = F.np.concatenate(field_proj_features, axis=-1)
            else:
                # TODO(sxjscience) May try to implement more advanced pooling methods for
                #  multimodal data.
                raise NotImplementedError
        scores = self.proj(agg_features)
        if len(self.out_shape) != 1:
            scores = F.np.reshape(scores, (-1,) + self.out_shape)
        return scores


@use_np
class BERTForTabularBasicV1(HybridBlock):
    """The basic model for tabular classification + regression with
    BERT (and its variants like ALBERT, MobileBERT, ELECTRA, etc.)
    as the backbone for handling text data.

    Here, we use the backbone network to extract the contextual embeddings and use
    another dense layer to map the contextual embeddings to the class scores.

    Input:

    TextField + EntityField --> TextNet -------> TextFeature
        ...
    CategoricalField --> CategoricalNet --> CategoricalFeature  ==> AggregateNet --> logits/scores
        ...
    NumericalField ----> NumericalNet ----> NumericalFeature
    """
    def __init__(self, text_backbone,
                 feature_field_info,
                 label_shape=None,
                 cfg=None,
                 prefix=None,
                 params=None):
        """

        Parameters
        ----------
        text_backbone
            Backbone network for handling the text data
        feature_field_info
            The field information of the training data. Each will be a tuple:
            - (field_type, attributes)
        label_shape
            The shape of the label/number of classes. If we need a scalar, it will be an empty tuple "()".
        cfg
            The configuration of the network
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        if cfg is None:
            cfg = BERTForTabularBasicV1.get_cfg()
        self.cfg = BERTForTabularBasicV1.get_cfg().clone_merge(cfg)
        assert self.cfg.TEXT_NET.pool_type == 'cls'
        feature_units = self.cfg.feature_units
        if feature_units == -1:
            feature_units = text_backbone.units
        if isinstance(label_shape, int):
            out_shape = (label_shape,)
        elif label_shape is None:
            out_shape = ()
        else:
            out_shape = label_shape
        with self.name_scope():
            self.text_backbone = text_backbone
            self.feature_field_info = feature_field_info
            self.categorical_fields = []
            self.numerical_fields = []
            self.agg_layer = FeatureAggregator(num_fields=len(feature_field_info),
                                               out_shape=out_shape,
                                               in_units=feature_units,
                                               cfg=cfg.AGG_NET)
            self.categorical_networks = nn.HybridSequential()
            self.numerical_networks = nn.HybridSequential()
            for i, (field_type_code, field_attrs) in enumerate(self.feature_field_info):
                if field_type_code == _C.CATEGORICAL:
                    with self.categorical_networks.name_scope():
                        self.categorical_networks.add(
                            CategoricalFeatureNet(num_class=field_attrs['prop'].num_class,
                                                  out_units=feature_units,
                                                  cfg=cfg.CATEGORICAL_NET))
                elif field_type_code == _C.NUMERICAL:
                    with self.numerical_networks.name_scope():
                        self.numerical_networks.add(
                            NumericalFeatureNet(input_shape=field_attrs['prop'].shape,
                                                out_units=feature_units))

    @staticmethod
    def get_cfg(key=None):
        if key is None:
            cfg = CfgNode()
            cfg.feature_units = -1  # -1 means not given and we will use the units of BERT
            # TODO(sxjscience) Use a class to store the TextNet
            cfg.TEXT_NET = CfgNode()
            cfg.TEXT_NET.use_segment_id = True
            cfg.TEXT_NET.pool_type = 'cls'
            cfg.AGG_NET = FeatureAggregator.get_cfg()
            cfg.CATEGORICAL_NET = CategoricalFeatureNet.get_cfg()
            cfg.NUMERICAL_NET = NumericalFeatureNet.get_cfg()
            cfg.INITIALIZER = CfgNode()
            cfg.INITIALIZER.weight = ['truncnorm', 0, 0.02]
            cfg.INITIALIZER.bias = ['zeros']
            return cfg
        else:
            raise NotImplementedError

    def initialize_with_pretrained_backbone(self, backbone_params_path, ctx=None, verbose=False):
        self.text_backbone.load_parameters(backbone_params_path, ctx=ctx)
        self.agg_layer.initialize(ctx=ctx)
        self.categorical_networks.initialize(ctx=ctx)
        self.numerical_networks.initialize(ctx=ctx)

    def hybrid_forward(self, F, features):
        """

        Parameters
        ----------
        features
            A list of field data

        Returns
        -------
        logits_or_scores
            Shape (batch_size,) + out_shape
        """
        field_features = []
        text_contextual_features = dict()
        categorical_count = 0
        numerical_count = 0
        for i, (field_type_code, field_attrs) in enumerate(self.feature_field_info):
            if field_type_code == _C.TEXT:
                batch_token_ids, batch_valid_length, batch_segment_ids, _ = features[i]
                if self.cfg.TEXT_NET.use_segment_id:
                    contextual_embedding, pooled_output = self.text_backbone(batch_token_ids,
                                                                             batch_segment_ids,
                                                                             batch_valid_length)
                else:
                    contextual_embedding = self.text_backbone(batch_token_ids, batch_valid_length)
                    pooled_output = contextual_embedding[:, 0, :]
                text_contextual_features[i] = contextual_embedding
                field_features.append(pooled_output)
            elif field_type_code == _C.ENTITY:
                # TODO Implement via segment-pool
                raise NotImplementedError('Currently not supported')
            elif field_type_code == _C.CATEGORICAL:
                batch_sample = features[i]
                extracted_feature = self.categorical_networks[categorical_count](batch_sample)
                categorical_count += 1
                field_features.append(extracted_feature)
            elif field_type_code == _C.NUMERICAL:
                batch_sample = features[i]
                extracted_feature = self.numerical_networks[numerical_count](batch_sample)
                numerical_count += 1
                field_features.append(extracted_feature)
        return self.agg_layer(field_features)
