# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Mobile BERT Model

@article{sun2020mobilebert,
  title={Mobilebert: a compact task-agnostic mobile bert for resource-limited devices},
  author={Sun, Zhiqing and Yu, Hongkun and Song, Xiaodan and Liu, Renjie and Yang, Yiming and Zhou, Denny},
  journal={arXiv preprint arXiv:2004.02984},
  year={2020}
}
"""


import os
from typing import Tuple, Optional

import mxnet as mx
import numpy as np
from mxnet import use_np
from mxnet.gluon import HybridBlock, nn

from ..op import select_vectors_by_position
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from ..layers import InitializerType, PositionwiseFFN, PositionalEmbedding, get_norm_layer, get_activation
from ..initializer import TruncNorm
from ..utils.config import CfgNode as CN
from ..utils.misc import load_checksum_stats, download
from ..registry import BACKBONE_REGISTRY
from ..attention_cell import MultiHeadAttentionCell, gen_self_attn_mask
from ..data.tokenizers import HuggingFaceWordPieceTokenizer

__all__ = ['MobileBertModel', 'MobileBertForMLM', 'MobileBertForPretrain',
           'list_pretrained_mobilebert', 'get_pretrained_mobilebert']

PRETRAINED_URL = {
    'google_uncased_mobilebert': {
        'cfg': 'google_uncased_mobilebert/model-1c33216b.yml',
        'vocab': 'google_uncased_mobilebert/vocab-e6d2b21d.json',
        'params': 'google_uncased_mobilebert/model-c8346cf2.params',
        'mlm_params': 'google_uncased_mobilebert/model_mlm-53948e82.params',
    }
}


FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'mobilebert.txt'))


@use_np
class MobileBertEncoderLayer(HybridBlock):
    """The Transformer Encoder Layer in Mobile Bert"""
    # TODO(zheyuye), use stacked groups for single ffn layer in transformer.TransformerEncoderLayer
    # and revise the other models and scripts, masking sure their are compatible.

    def __init__(self,
                 use_bottleneck: bool = True,
                 units: int = 512,
                 real_units: int = 128,
                 hidden_size: int = 2048,
                 num_heads: int = 8,
                 num_stacked_ffn: int = 1,
                 bottleneck_strategy: str = 'qk_sharing',
                 attention_dropout_prob: float = 0.1,
                 hidden_dropout_prob: float = 0.1,
                 activation_dropout_prob: float = 0.0,
                 activation: str = 'gelu',
                 normalization: str = 'layer_norm',
                 layer_norm_eps: float = 1e-12,
                 use_qkv_bias: bool = True,
                 weight_initializer: Optional[InitializerType] = None,
                 bias_initializer: Optional[InitializerType] = 'zeros',
                 dtype='float32',
                 prefix=None, params=None):
        """

        Parameters
        ----------
        use_bottleneck
        units
            size of inter-bottleneck
        real_units
            size of intra-bottleneck
        hidden_size
            size of feed-forward network
        num_heads
        num_stacked_ffn
        attention_dropout_prob
        hidden_dropout_prob
        activation_dropout_prob
        activation
        normalization
        layer_norm_eps
            onlyv valid when normalization is 'layer_norm'
        use_qkv_bias
        weight_initializer
        bias_initializer
        dtype
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        self._use_bottleneck = use_bottleneck
        self._units = units
        self._real_units = real_units
        self._num_heads = num_heads
        self._num_stacked_ffn = num_stacked_ffn
        self._bottleneck_strategy = bottleneck_strategy
        self._dtype = dtype
        assert real_units % num_heads == 0, 'units must be divisive by the number of heads'
        with self.name_scope():
            self.dropout_layer = nn.Dropout(hidden_dropout_prob)
            if use_bottleneck:
                self.in_bottleneck_proj = nn.Dense(units=real_units,
                                                   in_units=units,
                                                   flatten=False,
                                                   weight_initializer=weight_initializer,
                                                   bias_initializer=bias_initializer,
                                                   dtype=self._dtype,
                                                   prefix='in_bottleneck_proj_')
                self.in_bottleneck_ln = get_norm_layer(normalization=normalization,
                                                       in_channels=real_units,
                                                       epsilon=layer_norm_eps,
                                                       prefix='in_bottleneck_ln_')
                self.out_bottleneck_proj = nn.Dense(units=units,
                                                    in_units=real_units,
                                                    flatten=False,
                                                    weight_initializer=weight_initializer,
                                                    bias_initializer=bias_initializer,
                                                    dtype=self._dtype,
                                                    prefix='out_bottleneck_proj_')
                self.out_bottleneck_ln = get_norm_layer(normalization=normalization,
                                                        in_channels=units,
                                                        epsilon=layer_norm_eps,
                                                        prefix='out_bottleneck_ln_')

                if bottleneck_strategy == 'qk_sharing':
                    self.shared_qk = nn.Dense(units=real_units,
                                              in_units=units,
                                              flatten=False,
                                              weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer,
                                              dtype=self._dtype,
                                              prefix='shared_qk_')
                    self.shared_qk_ln = get_norm_layer(normalization=normalization,
                                                       in_channels=real_units,
                                                       epsilon=layer_norm_eps,
                                                       prefix='shared_qk_ln_')
            self.attention_proj = nn.Dense(units=real_units,
                                           flatten=False,
                                           in_units=real_units,
                                           use_bias=True,
                                           weight_initializer=weight_initializer,
                                           bias_initializer=bias_initializer,
                                           dtype=self._dtype,
                                           prefix='proj_')
            # The in_units of qkv varies according to the sharing strategy
            self.attn_query = nn.Dense(units=real_units,
                                       flatten=False,
                                       use_bias=use_qkv_bias,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       dtype=self._dtype,
                                       prefix='attn_query_')
            self.attn_key = nn.Dense(units=real_units,
                                     flatten=False,
                                     use_bias=use_qkv_bias,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer,
                                     dtype=self._dtype,
                                     prefix='attn_key_')
            self.attn_value = nn.Dense(units=real_units,
                                       flatten=False,
                                       use_bias=use_qkv_bias,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       dtype=self._dtype,
                                       prefix='attn_value_')
            self.attention_cell = \
                MultiHeadAttentionCell(
                    query_units=real_units,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout_prob,
                    scaled=True,
                    prefix='attn_cell_',
                    dtype=self._dtype,
                    layout='NTK'
                )
            self.layer_norm = get_norm_layer(normalization=normalization,
                                             in_channels=real_units,
                                             epsilon=layer_norm_eps,
                                             prefix='ln_')

            self.stacked_ffn = nn.HybridSequential(prefix='stacked_ffns_')
            with self.stacked_ffn.name_scope():
                for ffn_idx in range(num_stacked_ffn):
                    is_last_ffn = (ffn_idx == (num_stacked_ffn - 1))
                    # only apply dropout on last ffn layer if use bottleneck
                    dropout = float(hidden_dropout_prob * (not use_bottleneck) * is_last_ffn)
                    activation_dropout = float(activation_dropout_prob * (not use_bottleneck)
                                               * is_last_ffn)
                    self.stacked_ffn.add(
                        PositionwiseFFN(units=real_units,
                                        hidden_size=hidden_size,
                                        dropout=dropout,
                                        activation_dropout=activation_dropout,
                                        weight_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        activation=activation,
                                        normalization=normalization,
                                        layer_norm_eps=layer_norm_eps,
                                        dtype=self._dtype,
                                        prefix='{}_'.format(ffn_idx)))

    def hybrid_forward(self, F, data, attn_mask):
        """

        Parameters
        ----------
        F
        data :
            Shape (batch_size, seq_length, C_in)
        attn_mask :
            Shape (batch_size, seq_length, seq_length)

        Returns
        -------
        out :
            Shape (batch_size, seq_length, C_out)
        attn_weight :
            Shape (batch_size, seq_length, seq_length)
        """
        # TODO(sxjscience) Cannot use negative axis due to
        #  https://github.com/apache/incubator-mxnet/issues/18132
        if self._use_bottleneck:
            bn_proj = self.in_bottleneck_proj(data)
            bn_proj = self.in_bottleneck_ln(bn_proj)
            input = bn_proj
            if self._bottleneck_strategy == 'qk_sharing':
                # for Mobile Bert
                qk_shared = self.shared_qk(data)
                qk_shared = self.shared_qk_ln(qk_shared)
                query = qk_shared
                key = qk_shared
                value = data
            elif self._bottleneck_strategy == 'from_bottleneck':
                # for Mobile mobile bert Tiny
                query = bn_proj
                key = bn_proj
                value = bn_proj
            elif self._bottleneck_strategy == 'from_input':
                query = data
                key = data
                value = data
            else:
                raise NotImplementedError
        else:
            input = data
            query = data
            key = data
            value = data

        query = F.npx.reshape(self.attn_query(query), (-2, -2, self._num_heads, -1))
        key = F.npx.reshape(self.attn_key(key), (-2, -2, self._num_heads, -1))
        value = F.npx.reshape(self.attn_value(value), (-2, -2, self._num_heads, -1))
        out, [_, attn_weight] = self.attention_cell(query, key, value, attn_mask)
        out = self.attention_proj(out)
        if not self._use_bottleneck:
            out = self.dropout_layer(out)
        out = out + input
        out = self.layer_norm(out)
        for ffn_idx in range(self._num_stacked_ffn):
            ffn = self.stacked_ffn[ffn_idx]
            out = ffn(out)

        if self._use_bottleneck:
            out = self.out_bottleneck_proj(out)
            out = self.dropout_layer(out)
            out = out + data
            out = self.out_bottleneck_ln(out)
        return out, attn_weight


@use_np
class MobileBertTransformer(HybridBlock):
    def __init__(self,
                 use_bottleneck: bool = True,
                 units: int = 512,
                 hidden_size: int = 512,
                 inner_size: int = 128,
                 num_layers: int = 24,
                 num_heads: int = 4,
                 num_stacked_ffn: int = 1,
                 bottleneck_strategy: str = 'qk_sharing',
                 activation: str = 'gelu',
                 normalization: str = 'layer_norm',
                 attention_dropout_prob: float = 0.,
                 hidden_dropout_prob: float = 0.1,
                 output_attention: bool = False,
                 output_all_encodings: bool = False,
                 layer_norm_eps: float = 1E-12,
                 weight_initializer: InitializerType = TruncNorm(stdev=0.02),
                 bias_initializer: InitializerType = 'zeros',
                 dtype='float32',
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._dtype = dtype
        self._num_layers = num_layers
        self._output_attention = output_attention
        self._output_all_encodings = output_all_encodings

        assert bottleneck_strategy in ['qk_sharing', 'from_bottleneck', 'from_input'], \
            'The bottleneck strategy={} is not supported.'.format(bottleneck_strategy)
        real_units = inner_size if use_bottleneck else units
        assert real_units % num_heads == 0,\
            'In MobileBertTransformer, The real_units should be divided exactly ' \
            'by the number of heads. Received real_units={}, num_heads={}' \
            .format(real_units, num_heads)

        with self.name_scope():
            self.all_layers = nn.HybridSequential(prefix='layers_')
            with self.all_layers.name_scope():
                for layer_idx in range(num_layers):
                    self.all_layers.add(
                        MobileBertEncoderLayer(use_bottleneck=use_bottleneck,
                                               units=units,
                                               real_units=real_units,
                                               hidden_size=hidden_size,
                                               num_heads=num_heads,
                                               attention_dropout_prob=attention_dropout_prob,
                                               hidden_dropout_prob=hidden_dropout_prob,
                                               num_stacked_ffn=num_stacked_ffn,
                                               bottleneck_strategy=bottleneck_strategy,
                                               layer_norm_eps=layer_norm_eps,
                                               weight_initializer=weight_initializer,
                                               bias_initializer=bias_initializer,
                                               normalization=normalization,
                                               activation=activation,
                                               prefix='{}_'.format(layer_idx)))

    def hybrid_forward(self, F, data, valid_length):
        """
        Generate the representation given the inputs.

        This is used in training or fine-tuning a mobile bert model.

        Parameters
        ----------
        F
        data :
            Shape (batch_size, seq_length, C)
        valid_length :
            Shape (batch_size,)

        Returns
        -------
        out :
            Shape (batch_size, seq_length, C_out)
        """
        # 1. Embed the data
        attn_mask = gen_self_attn_mask(F, data, valid_length, dtype=self._dtype, attn_type='full')
        out = data
        all_encodings_outputs = []
        additional_outputs = []
        all_encodings_outputs.append(out)
        for layer_idx in range(self._num_layers):
            layer = self.all_layers[layer_idx]
            out, attention_weights = layer(out, attn_mask)
            # out : [batch_size, seq_len, units]
            # attention_weights : [batch_size, num_heads, seq_len, seq_len]
            if self._output_all_encodings:
                out = F.npx.sequence_mask(out,
                                          sequence_length=valid_length,
                                          use_sequence_length=True, axis=1)
                all_encodings_outputs.append(out)

            if self._output_attention:
                additional_outputs.append(attention_weights)

        if not self._output_all_encodings:
            # if self._output_all_encodings, SequenceMask is already applied above
            out = F.npx.sequence_mask(out, sequence_length=valid_length,
                                      use_sequence_length=True, axis=1)
            return out, additional_outputs
        else:
            return all_encodings_outputs, additional_outputs


@use_np
class MobileBertModel(HybridBlock):
    def __init__(self,
                 vocab_size: int = 30000,
                 embed_size: int = 128,
                 units: int = 512,
                 hidden_size: int = 512,
                 inner_size: int = 128,
                 max_length: int = 512,
                 num_heads: int = 4,
                 num_layers: int = 24,
                 num_stacked_ffn: int = 4,
                 bottleneck_strategy: str = 'qk_sharing',
                 activation: str = 'relu',
                 normalization: str = 'no_norm',
                 hidden_dropout_prob: int = 0.,
                 attention_dropout_prob: int = 0.1,
                 num_token_types: int = 2,
                 pos_embed_type: str = 'learned',
                 layer_norm_eps: float = 1E-12,
                 embed_initializer: InitializerType = TruncNorm(stdev=0.02),
                 weight_initializer: InitializerType = TruncNorm(stdev=0.02),
                 bias_initializer: InitializerType = 'zeros',
                 use_bottleneck=True,
                 trigram_embed=True,
                 use_pooler=True,
                 classifier_activation=False,
                 dtype='float32',
                 prefix=None,
                 params=None):
        super().__init__(prefix=prefix, params=params)
        self._dtype = dtype
        self.use_bottleneck = use_bottleneck
        self.bottleneck_strategy = bottleneck_strategy
        self.trigram_embed = trigram_embed
        self.normalization = normalization
        self.use_pooler = use_pooler
        self.classifier_activation = classifier_activation
        self.pos_embed_type = pos_embed_type
        self.num_token_types = num_token_types
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.units = units
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.activation = activation
        self.num_stacked_ffn = num_stacked_ffn
        self.embed_initializer = embed_initializer
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.layer_norm_eps = layer_norm_eps
        with self.name_scope():
            # Construct MobileBertTransformer
            self.encoder = MobileBertTransformer(
                units=units,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                inner_size=inner_size,
                num_stacked_ffn=num_stacked_ffn,
                bottleneck_strategy=bottleneck_strategy,
                attention_dropout_prob=attention_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                output_attention=False,
                output_all_encodings=False,
                activation=activation,
                normalization=normalization,
                layer_norm_eps=layer_norm_eps,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
                dtype=dtype,
                prefix='enc_',
            )
            self.encoder.hybridize()
            # Construct word embedding
            self.word_embed = nn.Embedding(input_dim=vocab_size,
                                           output_dim=embed_size,
                                           weight_initializer=embed_initializer,
                                           dtype=dtype,
                                           prefix='word_embed_')
            if trigram_embed or embed_size != units:
                self.embed_factorized_proj = nn.Dense(units=units,
                                                      flatten=False,
                                                      weight_initializer=weight_initializer,
                                                      bias_initializer=bias_initializer,
                                                      prefix='embed_factorized_proj_')
            self.embed_layer_norm = get_norm_layer(normalization=normalization,
                                                   in_channels=units,
                                                   epsilon=self.layer_norm_eps,
                                                   prefix='embed_ln_')

            self.embed_dropout = nn.Dropout(hidden_dropout_prob)
            # Construct token type embedding
            self.token_type_embed = nn.Embedding(input_dim=num_token_types,
                                                 output_dim=units,
                                                 weight_initializer=weight_initializer,
                                                 prefix='token_type_embed_')
            self.token_pos_embed = PositionalEmbedding(units=units,
                                                       max_length=max_length,
                                                       dtype=self._dtype,
                                                       method=pos_embed_type,
                                                       prefix='token_pos_embed_')
            if self.use_pooler and self.classifier_activation:
                # Construct pooler
                self.pooler = nn.Dense(units=units,
                                       in_units=units,
                                       flatten=False,
                                       activation='tanh',
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       prefix='pooler_')

    def hybrid_forward(self, F, inputs, token_types, valid_length):
        # pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a mobile bert model.

        Parameters
        ----------
        F
        inputs :
            Shape (batch_size, seq_length)
        token_types :
            Shape (batch_size, seq_length)

            If the inputs contain two sequences, we will set different token types for the first
             sentence and the second sentence.
        valid_length :
            The valid length of each sequence
            Shape (batch_size,)

        Returns
        -------
        contextual_embedding :
            Shape (batch_size, seq_length, units).
        pooled_output :
            This is optional. Shape (batch_size, units)
        """
        embedding = self.get_initial_embedding(F, inputs, token_types)

        contextual_embeddings, additional_outputs = self.encoder(embedding, valid_length)
        outputs = []
        outputs.append(contextual_embeddings)
        if self.use_pooler:
            pooled_out = self.apply_pooling(contextual_embeddings)
            outputs.append(pooled_out)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def get_initial_embedding(self, F, inputs, token_types=None, trigram_embed=True):
        """Get the initial token embeddings that considers the token type and positional embeddings

        Parameters
        ----------
        F
        inputs
            Shape (batch_size, seq_length)
        token_types
            Shape (batch_size, seq_length)
            If None, it will be initialized as all zero

        Returns
        -------
        embedding
            The initial embedding that will be fed into the encoder
        """
        word_embedding = self.word_embed(inputs)

        if trigram_embed:
            word_embedding = F.np.concatenate(
                [F.np.pad(word_embedding[:, 1:], ((0, 0), (0, 1), (0, 0))),
                 word_embedding,
                 F.np.pad(word_embedding[:, :-1], ((0, 0), (1, 0), (0, 0)))], axis=-1)
        # Projecting the embedding into units only for word embedding
        if trigram_embed or self.embed_size != self.units:
            embedding = self.embed_factorized_proj(word_embedding)

        if token_types is None:
            token_types = F.np.zeros_like(embedding)
        type_embedding = self.token_type_embed(token_types)
        embedding = embedding + type_embedding
        if self.pos_embed_type is not None:
            positional_embedding = self.token_pos_embed(F.npx.arange_like(embedding, axis=1))
            positional_embedding = F.np.expand_dims(positional_embedding, axis=0)
            embedding = embedding + positional_embedding
        # Extra layer normalization plus dropout
        embedding = self.embed_layer_norm(embedding)
        embedding = self.embed_dropout(embedding)
        return embedding

    def apply_pooling(self, sequence):
        """Generate the representation given the inputs.

        This is used for pre-training or fine-tuning a mobile bert model.
        Get the first token of the whole sequence which is [CLS]

        sequence:
            Shape (batch_size, sequence_length, units)
        return:
            Shape (batch_size, units)
        """
        outputs = sequence[:, 0, :]
        if self.classifier_activation:
            return self.pooler(outputs)
        else:
            return outputs

    @staticmethod
    def get_cfg(key=None):
        if key is None:
            cfg = CN()
            cfg.MODEL = CN()
            cfg.MODEL.vocab_size = 30522
            cfg.MODEL.embed_size = 128
            cfg.MODEL.units = 512
            cfg.MODEL.hidden_size = 512
            cfg.MODEL.inner_size = 128
            cfg.MODEL.max_length = 512
            cfg.MODEL.num_heads = 4
            cfg.MODEL.num_layers = 12
            cfg.MODEL.num_stacked_ffn = 4
            cfg.MODEL.pos_embed_type = 'learned'
            cfg.MODEL.activation = 'relu'
            cfg.MODEL.normalization = 'no_norm'
            cfg.MODEL.layer_norm_eps = 1E-12
            cfg.MODEL.bottleneck_strategy = 'qk_sharing'
            cfg.MODEL.num_token_types = 2
            cfg.MODEL.hidden_dropout_prob = 0.0
            cfg.MODEL.attention_dropout_prob = 0.1
            cfg.MODEL.dtype = 'float32'
            # Hyper-parameters of the Initializers
            cfg.INITIALIZER = CN()
            cfg.INITIALIZER.embed = ['truncnorm', 0, 0.02]
            cfg.INITIALIZER.weight = ['truncnorm', 0, 0.02]  # TruncNorm(0, 0.02)
            cfg.INITIALIZER.bias = ['zeros']
            # Version of the model. This helps ensure backward compatibility.
            # Also, we can not use string here due to https://github.com/rbgirshick/yacs/issues/26
            cfg.VERSION = 1
        else:
            raise NotImplementedError
        cfg.freeze()
        return cfg

    @classmethod
    def from_cfg(cls,
                 cfg,
                 use_bottleneck=True,
                 trigram_embed=True,
                 use_pooler=True,
                 classifier_activation=False,
                 prefix=None,
                 params=None):
        cfg = MobileBertModel.get_cfg().clone_merge(cfg)
        assert cfg.VERSION == 1, 'Wrong version!'
        embed_initializer = mx.init.create(*cfg.INITIALIZER.embed)
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
        return cls(vocab_size=cfg.MODEL.vocab_size,
                   units=cfg.MODEL.units,
                   hidden_size=cfg.MODEL.hidden_size,
                   embed_size=cfg.MODEL.embed_size,
                   num_layers=cfg.MODEL.num_layers,
                   num_heads=cfg.MODEL.num_heads,
                   bottleneck_strategy=cfg.MODEL.bottleneck_strategy,
                   inner_size=cfg.MODEL.inner_size,
                   num_stacked_ffn=cfg.MODEL.num_stacked_ffn,
                   max_length=cfg.MODEL.max_length,
                   hidden_dropout_prob=cfg.MODEL.hidden_dropout_prob,
                   attention_dropout_prob=cfg.MODEL.attention_dropout_prob,
                   num_token_types=cfg.MODEL.num_token_types,
                   pos_embed_type=cfg.MODEL.pos_embed_type,
                   activation=cfg.MODEL.activation,
                   normalization=cfg.MODEL.normalization,
                   layer_norm_eps=cfg.MODEL.layer_norm_eps,
                   dtype=cfg.MODEL.dtype,
                   embed_initializer=embed_initializer,
                   weight_initializer=weight_initializer,
                   bias_initializer=bias_initializer,
                   use_bottleneck=use_bottleneck,
                   trigram_embed=trigram_embed,
                   use_pooler=use_pooler,
                   classifier_activation=classifier_activation,
                   prefix=prefix,
                   params=params)


@use_np
class MobileBertForMLM(HybridBlock):
    def __init__(self, backbone_cfg,
                 use_bottleneck=True,
                 trigram_embed=True,
                 weight_initializer=None,
                 bias_initializer=None,
                 prefix=None,
                 params=None):
        """

        Parameters
        ----------
        backbone_cfg
        weight_initializer
        bias_initializer
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.backbone_model = MobileBertModel.from_cfg(backbone_cfg,
                                                           use_bottleneck=use_bottleneck,
                                                           trigram_embed=trigram_embed,
                                                           prefix='')
            if weight_initializer is None:
                weight_initializer = self.backbone_model.weight_initializer
            if bias_initializer is None:
                bias_initializer = self.backbone_model.bias_initializer
            self.mlm_decoder = nn.HybridSequential(prefix='mlm_')
            with self.mlm_decoder.name_scope():
                # Extra non-linear layer
                self.mlm_decoder.add(nn.Dense(units=self.backbone_model.units,
                                              flatten=False,
                                              weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer,
                                              prefix='proj_'))
                self.mlm_decoder.add(get_activation(self.backbone_model.activation))
                # use basic layer normalization for pretaining
                self.mlm_decoder.add(nn.LayerNorm(epsilon=self.backbone_model.layer_norm_eps,
                                                  prefix='ln_'))
            self.mlm_decoder.hybridize()
            # only load the dense weights with a re-initialized bias
            # parameters are stored in 'word_embed_bias' which is
            # not used in original embedding
            self.embedding_table = nn.Dense(
                units=self.backbone_model.vocab_size,
                in_units=self.backbone_model.embed_size,
                flatten=False,
                params=self.backbone_model.word_embed.collect_params('.*weight'),
                bias_initializer=bias_initializer,
                prefix='mlm_score_')
            if self.backbone_model.embed_size != self.backbone_model.units:
                self.extra_table = nn.Dense(
                    units=self.backbone_model.vocab_size,
                    use_bias=False,
                    in_units=self.backbone_model.units -
                    self.backbone_model.embed_size,
                    flatten=False,
                    prefix='mlm_extra_score_')

    def hybrid_forward(self, F, inputs, token_types, valid_length,
                       masked_positions):
        """Getting the scores of the masked positions.

        Parameters
        ----------
        F
        inputs :
            Shape (batch_size, seq_length)
        token_types :
            Shape (batch_size, seq_length)
            The type of the token. For example, if the inputs contain two sequences,
            we will set different token types for the first sentence and the second sentence.
        valid_length :
            The valid length of each sequence
            Shape (batch_size,)
        masked_positions :
            The masked position of the sequence
            Shape (batch_size, num_masked_positions).

        Returns
        -------
        contextual_embedding
            Shape (batch_size, seq_length, units).
        pooled_out
            Shape (batch_size, units)
        mlm_scores :
            Shape (batch_size, num_masked_positions, vocab_size)
        """
        contextual_embeddings, pooled_out = self.backbone_model(inputs, token_types, valid_length)
        mlm_features = select_vectors_by_position(F, contextual_embeddings, masked_positions)
        intermediate_output = self.mlm_decoder(mlm_features)
        if self.backbone_model.embed_size != self.backbone_model.units:
            scores = self.embedding_table(
                intermediate_output[:, :, :self.backbone_model.embed_size])
            extra_scores = self.extra_table(
                intermediate_output[:, :, self.backbone_model.embed_size:])
            mlm_scores = scores + extra_scores
        else:
            mlm_scores = self.embedding_table(intermediate_output)
        return contextual_embeddings, pooled_out, mlm_scores


@use_np
class MobileBertForPretrain(HybridBlock):
    def __init__(self, backbone_cfg,
                 use_bottleneck=True,
                 trigram_embed=True,
                 weight_initializer=None,
                 bias_initializer=None,
                 prefix=None, params=None):
        """

        Parameters
        ----------
        backbone_cfg
            The cfg of the backbone model
        weight_initializer
        bias_initializer
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.backbone_model = MobileBertModel.from_cfg(backbone_cfg,
                                                           use_bottleneck=use_bottleneck,
                                                           trigram_embed=trigram_embed,
                                                           prefix='')
            if weight_initializer is None:
                weight_initializer = self.backbone_model.weight_initializer
            if bias_initializer is None:
                bias_initializer = self.backbone_model.bias_initializer
            # Construct nsp_classifier for next sentence prediction
            self.nsp_classifier = nn.Dense(units=2,
                                           weight_initializer=weight_initializer,
                                           prefix='nsp_')
            self.mlm_decoder = nn.HybridSequential(prefix='mlm_')
            with self.mlm_decoder.name_scope():
                # Extra non-linear layer
                self.mlm_decoder.add(nn.Dense(units=self.backbone_model.units,
                                              flatten=False,
                                              weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer,
                                              prefix='proj_'))
                self.mlm_decoder.add(get_activation(self.backbone_model.activation))
                # use basic layer normalization for pretaining
                self.mlm_decoder.add(nn.LayerNorm(epsilon=self.backbone_model.layer_norm_eps,
                                                  prefix='ln_'))
            self.mlm_decoder.hybridize()
            # only load the dense weights with a re-initialized bias
            # parameters are stored in 'word_embed_bias' which is
            # not used in original embedding
            self.embedding_table = nn.Dense(
                units=self.backbone_model.vocab_size,
                in_units=self.backbone_model.embed_size,
                flatten=False,
                params=self.backbone_model.word_embed.collect_params('.*weight'),
                bias_initializer=bias_initializer,
                prefix='mlm_score_')
            if self.backbone_model.embed_size != self.backbone_model.units:
                self.extra_table = nn.Dense(
                    units=self.backbone_model.vocab_size,
                    in_units=self.backbone_model.units -
                    self.backbone_model.embed_size,
                    flatten=False,
                    use_bias=False,
                    bias_initializer=bias_initializer,
                    prefix='mlm_extra_score_')

    def hybrid_forward(self, F, inputs, token_types, valid_length,
                       masked_positions):
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a mobile mobile bert model.

        Parameters
        ----------
        F
        inputs :
            Shape (batch_size, seq_length)
        token_types :
            Shape (batch_size, seq_length)

            If the inputs contain two sequences, we will set different token types for the first
             sentence and the second sentence.
        valid_length :
            The valid length of each sequence
            Shape (batch_size,)
        masked_positions :
            The masked position of the sequence
            Shape (batch_size, num_masked_positions).

        Returns
        -------
        contextual_embedding
            Shape (batch_size, seq_length, units).
        pooled_out
            Shape (batch_size, units)
        nsp_score :
            Shape (batch_size, 2)
        mlm_scores :
            Shape (batch_size, num_masked_positions, vocab_size)
        """
        contextual_embeddings, pooled_out = self.backbone_model(inputs, token_types, valid_length)
        nsp_score = self.nsp_classifier(pooled_out)
        mlm_features = select_vectors_by_position(F, contextual_embeddings, masked_positions)
        intermediate_output = self.mlm_decoder(mlm_features)
        if self.backbone_model.embed_size != self.backbone_model.units:
            scores = self.embedding_table(
                intermediate_output[:, :, :self.backbone_model.embed_size])
            extra_scores = self.extra_table(
                intermediate_output[:, :, self.backbone_model.embed_size:])
            mlm_scores = scores + extra_scores
        else:
            mlm_scores = self.embedding_table(intermediate_output)
        return contextual_embeddings, pooled_out, nsp_score, mlm_scores


def list_pretrained_mobilebert():
    return sorted(list(PRETRAINED_URL.keys()))


def get_pretrained_mobilebert(model_name: str = 'google_uncased_mobilebert',
                              root: str = get_model_zoo_home_dir(),
                              load_backbone=True, load_mlm=True)\
        -> Tuple[CN, HuggingFaceWordPieceTokenizer, str, str]:
    """Get the pretrained mobile bert weights

    Parameters
    ----------
    model_name
        The name of the mobile bert model.
    root
        The downloading root
    load_backbone
        Whether to load the weights of the backbone network
    load_mlm
        Whether to load the weights of MLM

    Returns
    -------
    cfg
        Network configuration
    tokenizer
        The HuggingFaceWordPieceTokenizer
    backbone_params_path
        Path to the parameter of the backbone network
    mlm_params_path
        Path to the parameter that includes both the backbone and the MLM
    """
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}'.format(
        model_name, list_pretrained_mobilebert())
    cfg_path = PRETRAINED_URL[model_name]['cfg']
    vocab_path = PRETRAINED_URL[model_name]['vocab']
    params_path = PRETRAINED_URL[model_name]['params']
    mlm_params_path = PRETRAINED_URL[model_name]['mlm_params']
    local_paths = dict()
    for k, path in [('cfg', cfg_path), ('vocab', vocab_path)]:
        local_paths[k] = download(url=get_repo_model_zoo_url() + path,
                                  path=os.path.join(root, path),
                                  sha1_hash=FILE_STATS[path])
    if load_backbone:
        local_params_path = download(url=get_repo_model_zoo_url() + params_path,
                                     path=os.path.join(root, params_path),
                                     sha1_hash=FILE_STATS[params_path])
    else:
        local_params_path = None
    if load_mlm and mlm_params_path is not None:
        local_mlm_params_path = download(url=get_repo_model_zoo_url() + mlm_params_path,
                                         path=os.path.join(root, mlm_params_path),
                                         sha1_hash=FILE_STATS[mlm_params_path])
    else:
        local_mlm_params_path = None
    do_lower = True if 'lowercase' in PRETRAINED_URL[model_name]\
                       and PRETRAINED_URL[model_name]['lowercase'] else False
    # TODO(sxjscience) Move do_lower to assets.
    tokenizer = HuggingFaceWordPieceTokenizer(
                    vocab_file=local_paths['vocab'],
                    unk_token='[UNK]',
                    pad_token='[PAD]',
                    cls_token='[CLS]',
                    sep_token='[SEP]',
                    mask_token='[MASK]',
                    lowercase=do_lower)
    cfg = MobileBertModel.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_params_path, local_mlm_params_path


BACKBONE_REGISTRY.register('mobilebert', [MobileBertModel,
                                          get_pretrained_mobilebert,
                                          list_pretrained_mobilebert])
