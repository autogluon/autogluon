import numpy as np
import mxnet as mx
from mxnet import use_np
from mxnet.gluon import nn, HybridBlock
from typing import Optional, Tuple, List
from ..utils.registry import Registry
from ..attention_cell import MultiHeadAttentionCell, gen_self_attn_mask, gen_mem_attn_mask
from ..layers import PositionalEmbedding, PositionwiseFFN, InitializerType
from ..utils.config import CfgNode as CN
from ..sequence_sampler import BaseStepDecoder
__all__ = ['TransformerEncoderLayer', 'TransformerDecoderLayer',
           'TransformerEncoder', 'TransformerDecoder',
           'TransformerNMTModel', 'TransformerNMTInference']

transformer_nmt_cfg_reg = Registry('transformer_nmt_cfg')


@transformer_nmt_cfg_reg.register()
def transformer_nmt_base():
    """Configuration of Transformer WMT EN-DE Base"""
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.src_vocab_size = -1
    cfg.MODEL.tgt_vocab_size = -1
    cfg.MODEL.max_src_length = -1
    cfg.MODEL.max_tgt_length = -1
    cfg.MODEL.scale_embed = True
    cfg.MODEL.pos_embed_type = "sinusoidal"
    cfg.MODEL.shared_embed = True
    cfg.MODEL.tie_weights = True
    cfg.MODEL.attention_dropout = 0.0
    cfg.MODEL.activation_dropout = 0.0
    cfg.MODEL.dropout = 0.1

    # Parameters for the encoder
    cfg.MODEL.ENCODER = CN()
    cfg.MODEL.ENCODER.num_layers = 6
    cfg.MODEL.ENCODER.units = 512
    cfg.MODEL.ENCODER.num_heads = 8
    cfg.MODEL.ENCODER.hidden_size = 2048
    cfg.MODEL.ENCODER.recurrent = False
    cfg.MODEL.ENCODER.activation = 'relu'
    cfg.MODEL.ENCODER.pre_norm = False

    # Parameters for the decoder
    cfg.MODEL.DECODER = CN()
    cfg.MODEL.DECODER.num_layers = 6
    cfg.MODEL.DECODER.units = 512
    cfg.MODEL.DECODER.num_heads = 8
    cfg.MODEL.DECODER.hidden_size = 2048
    cfg.MODEL.DECODER.recurrent = False
    cfg.MODEL.DECODER.activation = 'relu'
    cfg.MODEL.DECODER.pre_norm = False

    # Parameters for mixture of models
    cfg.MODEL.method = 'hMoElp'
    cfg.MODEL.num_experts = 3

    # Parameters for the initializer
    cfg.INITIALIZER = CN()
    cfg.INITIALIZER.embed = ['xavier', 'gaussian', 'in', 1.0]
    cfg.INITIALIZER.weight = ['xavier', 'uniform', 'avg', 3.0]
    cfg.INITIALIZER.bias = ['zeros']
    cfg.VERSION = 1
    cfg.freeze()
    return cfg


@transformer_nmt_cfg_reg.register()
def transformer_nmt_base_prenorm():
    cfg = transformer_nmt_base()
    cfg.defrost()
    cfg.MODEL.ENCODER.pre_norm = True
    cfg.MODEL.DECODER.pre_norm = True
    cfg.freeze()
    return cfg


@transformer_nmt_cfg_reg.register()
def transformer_iwslt_de_en():
    cfg = TransformerNMTModel.get_cfg()
    cfg.defrost()
    cfg.MODEL.ENCODER.units = 512
    cfg.MODEL.ENCODER.hidden_size = 1024
    cfg.MODEL.ENCODER.num_heads = 4
    cfg.MODEL.ENCODER.num_layers = 6
    cfg.MODEL.DECODER.units = 512
    cfg.MODEL.DECODER.hidden_size = 1024
    cfg.MODEL.DECODER.num_heads = 4
    cfg.MODEL.DECODER.num_layers = 6
    cfg.freeze()
    return cfg


@transformer_nmt_cfg_reg.register()
def transformer_wmt_en_de_big():
    """Same wmt_en_de_big architecture as in FairSeq"""
    cfg = TransformerNMTModel.get_cfg()
    cfg.defrost()
    cfg.MODEL.attention_dropout = 0.1
    cfg.MODEL.dropout = 0.3
    cfg.MODEL.ENCODER.units = 1024
    cfg.MODEL.ENCODER.hidden_size = 4096
    cfg.MODEL.ENCODER.num_heads = 16
    cfg.MODEL.ENCODER.num_layers = 6
    cfg.MODEL.DECODER.units = 1024
    cfg.MODEL.DECODER.hidden_size = 4096
    cfg.MODEL.DECODER.num_heads = 16
    cfg.MODEL.DECODER.num_layers = 6
    cfg.freeze()
    return cfg


@transformer_nmt_cfg_reg.register()
def transformer_wmt_en_de_big_t2t():
    """Parameter used in the T2T implementation"""
    cfg = transformer_wmt_en_de_big()
    cfg.defrost()
    cfg.MODEL.attention_dropout = 0.1
    cfg.MODEL.activation_dropout = 0.1
    cfg.MODEL.ENCODER.pre_norm = True
    cfg.MODEL.DECODER.pre_norm = True
    cfg.freeze()
    return cfg


@use_np
class TransformerEncoderLayer(HybridBlock):
    """Transformer Encoder Layer"""
    def __init__(self,
                 units: int = 512,
                 hidden_size: int = 2048,
                 num_heads: int = 8,
                 attention_dropout_prob: float = 0.1,
                 hidden_dropout_prob: float = 0.1,
                 activation_dropout_prob: float = 0.0,
                 layer_norm_eps: float = 1e-12,
                 pre_norm: bool = False,
                 use_qkv_bias: bool = True,
                 weight_initializer: Optional[InitializerType] = None,
                 bias_initializer: Optional[InitializerType] = 'zeros',
                 activation: str = 'relu',
                 dtype='float32',
                 prefix=None, params=None):
        """

        Parameters
        ----------
        units
        hidden_size
        num_heads
        attention_dropout_prob
        hidden_dropout_prob
        activation_dropout_prob
        layer_norm_eps
        pre_norm
            Whether to attach the normalization layer before attention layer
            If pre_norm:
                norm(data) -> attn -> res(+data) -> ffn
            Else:
                data -> attn -> norm(res(+data)) -> ffn

        use_qkv_bias
        weight_initializer
        bias_initializer
        activation
        dtype
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        self._units = units
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._attention_dropout_prob = attention_dropout_prob
        self._hidden_dropout_prob = hidden_dropout_prob
        self._activation_dropout_prob = activation_dropout_prob
        self._pre_norm = pre_norm
        self._dtype = dtype
        assert self._units % self._num_heads == 0, 'units must be divisive by the number of heads'
        with self.name_scope():
            self.dropout_layer = nn.Dropout(hidden_dropout_prob)
            self.attn_qkv = nn.Dense(3 * units,
                                     flatten=False,
                                     use_bias=use_qkv_bias,
                                     in_units=units,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer,
                                     dtype=self._dtype,
                                     prefix='attn_qkv_')
            self.attention_proj = nn.Dense(units=units,
                                           flatten=False,
                                           in_units=units,
                                           use_bias=True,
                                           weight_initializer=weight_initializer,
                                           bias_initializer=bias_initializer,
                                           dtype=self._dtype,
                                           prefix='proj_')
            self.attention_cell =\
                MultiHeadAttentionCell(
                    query_units=self._units,
                    num_heads=self._num_heads,
                    attention_dropout=self._attention_dropout_prob,
                    scaled=True,
                    prefix='attn_cell_',
                    dtype=self._dtype,
                    layout='NTK'
                )
            self.layer_norm = nn.LayerNorm(epsilon=layer_norm_eps,
                                           in_channels=units,
                                           prefix='ln_')
            self.ffn = PositionwiseFFN(units=units,
                                       hidden_size=hidden_size,
                                       dropout=hidden_dropout_prob,
                                       activation_dropout=activation_dropout_prob,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       layer_norm_eps=layer_norm_eps,
                                       activation=activation,
                                       pre_norm=pre_norm,
                                       dtype=self._dtype,
                                       prefix='ffn_')

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
        if self._pre_norm:
            data = self.layer_norm(data)
        query, key, value = F.np.split(self.attn_qkv(data), 3, axis=-1)
        query = F.npx.reshape(query, (-2, -2, self._num_heads, -1))
        key = F.npx.reshape(key, (-2, -2, self._num_heads, -1))
        value = F.npx.reshape(value, (-2, -2, self._num_heads, -1))
        out, [_, attn_weight] = self.attention_cell(query, key, value, attn_mask)
        out = self.attention_proj(out)
        out = self.dropout_layer(out)
        out = out + data
        if not self._pre_norm:
            out = self.layer_norm(out)
        out = self.ffn(out)
        return out, attn_weight


@use_np
class TransformerEncoder(HybridBlock):
    def __init__(self, num_layers=6, recurrent=False,
                 units=512, hidden_size=2048, num_heads=8,
                 activation_dropout=0.0, dropout=0.1,
                 attention_dropout=0.1, layer_norm_eps=1E-5, data_norm=False,
                 pre_norm=False, weight_initializer=None, bias_initializer='zeros',
                 activation='relu', dtype='float32',
                 prefix=None, params=None):
        """

        Parameters
        ----------
        num_layers :
            The number of layers
        recurrent : bool
            Whether the layers share weights or not
        units
        hidden_size
        num_heads
        dropout
        layer_norm_eps
        data_norm
            Whether to apply LayerNorm to the data
        pre_norm
            Whether to apply LayerNorm before the attention layer.
        weight_initializer
        bias_initializer
        activation
        prefix
        params
        """
        super(TransformerEncoder, self).__init__(prefix=prefix, params=params)
        self._dtype = dtype
        self.num_layers = num_layers
        self._recurrent = recurrent
        self._data_norm = data_norm
        self._pre_norm = pre_norm
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            if self._pre_norm:
                self.ln_final = nn.LayerNorm(epsilon=layer_norm_eps,
                                             in_channels=units,
                                             prefix='ln_final_')
            if self._data_norm:
                self.ln_data = nn.LayerNorm(epsilon=layer_norm_eps,
                                            in_channels=units,
                                            prefix='ln_data_')
            # Construct the intermediate layers
            self.layers = nn.HybridSequential(prefix='layers_')
            real_num_layers = 1 if recurrent else num_layers
            with self.layers.name_scope():
                for i in range(real_num_layers):
                    self.layers.add(TransformerEncoderLayer(
                        units=units,
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        hidden_dropout_prob=dropout,
                        attention_dropout_prob=attention_dropout,
                        activation_dropout_prob=activation_dropout,
                        layer_norm_eps=layer_norm_eps,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer,
                        pre_norm=pre_norm,
                        activation=activation,
                        dtype=dtype,
                        prefix='{}_'.format(i)))

    def hybrid_forward(self, F, data, valid_length):
        """

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
        attn_mask = gen_self_attn_mask(F, data, valid_length,
                                       dtype=self._dtype, attn_type='full')
        out = self.dropout_layer(data)
        if self._data_norm:
            out = self.ln_data(out)
        for i in range(self.num_layers):
            if self._recurrent:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            out, _ = layer(out, attn_mask)
        if self._pre_norm:
            out = self.ln_final(out)
        return out


@use_np
class TransformerDecoderLayer(HybridBlock):
    def __init__(self, units: int = 512,
                 mem_units: Optional[int] = None,
                 hidden_size: int = 2048,
                 num_heads: int = 8,
                 activation_dropout: float = 0.0,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 layer_norm_eps: float = 1E-5,
                 activation: str = 'relu',
                 pre_norm: bool = False,
                 weight_initializer=None,
                 bias_initializer='zeros',
                 dtype='float32',
                 prefix=None,
                 params=None):
        """

        Parameters
        ----------
        units
        mem_units
            The number of units in the memory. By default, it is initialized to be the
            same as the units.
        hidden_size
        num_heads
        activation_dropout
        dropout
        attention_dropout
        layer_norm_eps
        activation
        pre_norm
            Whether to apply normalization before the attention layer
        weight_initializer
        bias_initializer
        dtype
        prefix
        params
        """
        super(TransformerDecoderLayer, self).__init__(prefix=prefix, params=params)
        self._dtype = dtype
        self._units = units
        if mem_units is None:
            mem_units = units
        self._mem_units = mem_units
        self._pre_norm = pre_norm
        self._num_heads = num_heads
        self._attention_dropout = attention_dropout
        self._dtype = dtype
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            if units % num_heads:
                raise ValueError('In Transformer, units should be divided exactly by the number of '
                                 'heads. Received units={}, num_heads={}'.format(units, num_heads))
            self.attn_in_qkv = nn.Dense(3 * units, in_units=units,
                                        use_bias=False,
                                        flatten=False,
                                        weight_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        dtype=dtype,
                                        prefix='attn_in_qkv_')
            self.self_attention = MultiHeadAttentionCell(query_units=units,
                                                         num_heads=num_heads,
                                                         attention_dropout=self._attention_dropout,
                                                         dtype=dtype,
                                                         layout='NTK',
                                                         prefix='self_attn_')
            self.proj_in = nn.Dense(units=units, in_units=units, flatten=False,  use_bias=False,
                                    weight_initializer=weight_initializer,
                                    bias_initializer=bias_initializer,
                                    dtype=dtype,
                                    prefix='proj_in_')
            self.attn_inter_q = nn.Dense(units,
                                         in_units=units,
                                         use_bias=False,
                                         flatten=False,
                                         weight_initializer=weight_initializer,
                                         bias_initializer=bias_initializer,
                                         dtype=dtype,
                                         prefix='attn_inter_q_')
            self.attn_inter_k = nn.Dense(units, in_units=mem_units,
                                         use_bias=False,
                                         flatten=False,
                                         weight_initializer=weight_initializer,
                                         bias_initializer=bias_initializer,
                                         dtype=dtype,
                                         prefix='attn_inter_k_')
            self.attn_inter_v = nn.Dense(units, in_units=mem_units,
                                         use_bias=False,
                                         flatten=False,
                                         weight_initializer=weight_initializer,
                                         bias_initializer=bias_initializer,
                                         dtype=dtype,
                                         prefix='attn_inter_v_')
            self.inter_attention = MultiHeadAttentionCell(query_units=units,
                                                          num_heads=num_heads,
                                                          attention_dropout=self._attention_dropout,
                                                          dtype=dtype,
                                                          layout='NTK',
                                                          prefix='inter_attn_')
            self.proj_inter = nn.Dense(units=units, in_units=units,
                                       flatten=False, use_bias=False,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       dtype=dtype,
                                       prefix='proj_inter_')
            # TODO(sxjscience) Add DType to LayerNorm
            self.ln_in = nn.LayerNorm(epsilon=layer_norm_eps,
                                      in_channels=units,
                                      prefix='ln_in_')
            self.ln_inter = nn.LayerNorm(epsilon=layer_norm_eps,
                                         in_channels=units,
                                         prefix='ln_inter_')
            self.ffn = PositionwiseFFN(units=units,
                                       hidden_size=hidden_size,
                                       dropout=dropout,
                                       activation_dropout=activation_dropout,
                                       activation=activation,
                                       pre_norm=pre_norm,
                                       dtype=dtype,
                                       prefix='ffn_')

    def hybrid_forward(self, F, data, mem, self_causal_mask, mem_attn_mask):
        """

        Parameters
        ----------
        F
        data :
            Shape (batch_size, seq_length, C_in)
        mem :
            Shape (batch_size, mem_length, C_mem)
        self_causal_mask :
            Shape (batch_size, seq_length, seq_length)
            Mask for the causal self-attention.
            self_causal_mask[i, j, :] masks the elements that token `j` attends to.
            To understand the self-causal attention mask, we can look at the following example:
                       ['I', 'can', 'now', 'use', 'numpy', 'in', 'Gluon@@', 'NLP']
            'I':         1,    0,     0,     0,      0,     0,      0,      0
            'can':       1,    1,     0,     0,      0,     0,      0,      0
            'now':       1,    1,     1,     0,      0,     0,      0,      0
            'use':       1,    1,     1,     1,      0,     0,      0,      0
            'numpy':     1,    1,     1,     1,      1,     0,      0,      0
            'in':        1,    1,     1,     1,      1,     1,      0,      0
            'Gluon@@':   1,    1,     1,     1,      1,     1,      1,      0
            'NLP':       1,    1,     1,     1,      1,     1,      1,      1
        mem_attn_mask :
            Shape (batch_size, seq_length, mem_length)
            Mask between the decoding input and the memory.
                       ['numpy', 'in', 'Gluon@@', 'NLP']
            'I':         1,     1,      1,      1
            'can':       1,     1,      1,      1
            'now':       1,     1,      1,      1
            'use':       1,     1,      1,      1

        Returns
        -------
        out :
            Shape (batch_size, seq_length, C_out)
        """
        # TODO(szhengac)
        #  Try the architecture in the "[ECCV2016] Identity Mappings in Deep Residual Networks".
        #  Shuai proposed to switch the order of the activation layer.
        # 1. Get the causal self-attention value
        if self._pre_norm:
            data = self.ln_in(data)
        self_query, self_key, self_value = F.np.split(self.attn_in_qkv(data), 3, axis=-1)
        out, _ = self.self_attention(F.npx.reshape(self_query, (-2, -2, self._num_heads, -1)),
                                     F.npx.reshape(self_key, (-2, -2, self._num_heads, -1)),
                                     F.npx.reshape(self_value, (-2, -2, self._num_heads, -1)),
                                     self_causal_mask)
        out = self.proj_in(out)
        out = self.dropout_layer(out)
        out = out + data
        if not self._pre_norm:
            out = self.ln_in(out)
        # 2. Attend to the contextual memory
        data = out
        if self._pre_norm:
            data = self.ln_inter(data)
        out, _ = self.inter_attention(F.npx.reshape(self.attn_inter_q(data),
                                                    (-2, -2, self._num_heads, -1)),
                                      F.npx.reshape(self.attn_inter_k(mem),
                                                    (-2, -2, self._num_heads, -1)),
                                      F.npx.reshape(self.attn_inter_v(mem),
                                                    (-2, -2, self._num_heads, -1)),
                                      mem_attn_mask)
        out = self.proj_inter(out)
        out = self.dropout_layer(out)
        out = out + data
        if not self._pre_norm:
            out = self.ln_inter(out)
        # 3. Encode the output via an FFN layer
        out = self.ffn(out)
        return out

    @property
    def state_batch_axis(self):
        return 0, 0

    def init_states(self, batch_size, ctx, dtype='float32'):
        """Initialize the states required for incremental decoding

        Returns
        -------
        init_key :
            Shape (batch_size, 0, N, C_key)
        init_value :
            Shape (batch_size, 0, N, C_value)
        """
        init_key = mx.np.zeros(shape=(batch_size, 0, self._num_heads,
                                      self._units // self._num_heads), ctx=ctx, dtype=dtype)
        init_value = mx.np.zeros(shape=(batch_size, 0, self._num_heads,
                                        self._units // self._num_heads), ctx=ctx, dtype=dtype)
        return init_key, init_value

    def incremental_decode(self, F, data, states, mem, mem_valid_length, mem_attn_mask=None):
        """Incrementally generate the output given the decoder input.

        Parameters
        ----------
        F
        data
            Shape (batch_size, 1, C_in)
        states
            The previous states, contains
            - prev_multi_key
                Shape (batch_size, prev_seq_length, num_heads, C_key)
            - prev_multi_value
                Shape (batch_size, prev_seq_length, num_heads, C_value)
        mem
            The memory
            Shape (batch_size, mem_length, C_mem)
        mem_valid_length
            Valid length of the memory
            Shape (batch_size,)
        mem_attn_mask
            The attention mask between data and the memory
            Has shape (batch_size, 1, mem_length)

        Returns
        -------
        out
            Shape (batch_size, 1, C_out)
        updated_states
            - new_key
                Shape (batch_size, prev_seq_length + 1, num_heads, C_key)
            - new_value
                Shape (batch_size, prev_seq_length + 1, num_heads, C_value)
        """
        if self._pre_norm:
            data = self.ln_in(data)
        prev_key, prev_value = states  # Shape (B, prev_L, #Head, C_K), (B, prev_L, #Head, C_V)
        if mem_attn_mask is None:
            mem_attn_mask = gen_mem_attn_mask(F, mem, mem_valid_length, data, None,
                                              dtype=self._dtype)
        # 1. Get the causal self-attention value, we need to attend to both the current data
        # and the previous stored key/values
        step_qkv = self.attn_in_qkv(data)  # Shape (B, 1, 3 * num_heads * C_key)
        step_query, step_key, step_value = F.np.split(step_qkv, 3, axis=-1)
        step_query = F.npx.reshape(step_query, (-2, -2, self._num_heads, -1))
        step_key = F.npx.reshape(step_key, (-2, -2, self._num_heads, -1))
        step_value = F.npx.reshape(step_value, (-2, -2, self._num_heads, -1))
        new_key = F.np.concatenate([prev_key, step_key], axis=1)
        new_value = F.np.concatenate([prev_value, step_value], axis=1)
        out, _ = self.self_attention(step_query, new_key, new_value, None)
        out = self.proj_in(out)
        out = self.dropout_layer(out)
        out = out + data
        if not self._pre_norm:
            out = self.ln_in(out)
        # 2. Attend to the contextual memory
        data = out
        if self._pre_norm:
            data = self.ln_inter(data)
        out, _ = self.inter_attention(F.npx.reshape(self.attn_inter_q(data),
                                                    (-2, -2, self._num_heads, -1)),
                                      F.npx.reshape(self.attn_inter_k(mem),
                                                    (-2, -2, self._num_heads, -1)),
                                      F.npx.reshape(self.attn_inter_v(mem),
                                                    (-2, -2, self._num_heads, -1)),
                                      mem_attn_mask)
        out = self.proj_inter(out)
        out = self.dropout_layer(out)
        out = out + data
        if not self._pre_norm:
            out = self.ln_inter(out)
        # 3. Encode the output via an FFN layer
        out = self.ffn(out)
        return out, (new_key, new_value)


@use_np
class TransformerDecoder(HybridBlock):
    def __init__(self, num_layers=6, recurrent=False,
                 units=512, mem_units=None, hidden_size=2048,
                 num_heads=8, max_shift=None, rel_pos_embed=False, activation_dropout=0.0,
                 dropout=0.1, attention_dropout=0.1, layer_norm_eps=1E-5, data_norm=False,
                 pre_norm=False, weight_initializer=None, bias_initializer=None,
                 activation='relu', dtype='float32', prefix=None, params=None):
        super(TransformerDecoder, self).__init__(prefix=prefix, params=params)
        self._dtype = dtype
        self._units = units
        self._mem_units = mem_units
        self.num_layers = num_layers
        self.recurrent = recurrent
        self.max_shift = max_shift
        self.rel_pos_embed = rel_pos_embed
        self._data_norm = data_norm
        self._pre_norm = pre_norm
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            if self._data_norm:
                self.ln_data = nn.LayerNorm(epsilon=layer_norm_eps,
                                            in_channels=units,
                                            prefix='ln_data_')
            if self._pre_norm:
                self.ln_final = nn.LayerNorm(epsilon=layer_norm_eps,
                                             in_channels=units,
                                             prefix='ln_final_')
            # Construct the intermediate layers
            self.layers = nn.HybridSequential(prefix='layers_')
            real_num_layers = 1 if recurrent else num_layers
            with self.layers.name_scope():
                for i in range(real_num_layers):
                    self.layers.add(TransformerDecoderLayer(units=units,
                                                            mem_units=mem_units,
                                                            hidden_size=hidden_size,
                                                            num_heads=num_heads,
                                                            activation_dropout=activation_dropout,
                                                            dropout=dropout,
                                                            attention_dropout=attention_dropout,
                                                            layer_norm_eps=layer_norm_eps,
                                                            weight_initializer=weight_initializer,
                                                            bias_initializer=bias_initializer,
                                                            activation=activation,
                                                            pre_norm=pre_norm,
                                                            dtype=dtype,
                                                            prefix='{}_'.format(i)))

    def hybrid_forward(self, F, data, valid_length, mem_data, mem_valid_length):
        """

        Parameters
        ----------
        F
        data :
            Shape (batch_size, seq_length, C_in)
        valid_length :
            Shape (batch_size,)
        mem_data :
            Shape (batch_size, mem_length, C_mem)
        mem_valid_length :
            Shape (batch_size,)
        Returns
        -------
        out :
            Shape (batch_size, seq_length, C_out)
        """
        # 1. Embed the data
        out = self.dropout_layer(data)
        if self._data_norm:
            out = self.ln_data(out)
        self_causal_mask = gen_self_attn_mask(F, data, valid_length,
                                              dtype=self._dtype, attn_type='causal')
        mem_attn_mask = gen_mem_attn_mask(F, mem_data, mem_valid_length, data, valid_length,
                                          dtype=self._dtype)
        for i in range(self.num_layers):
            if self.recurrent:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            out = layer(out, mem_data, self_causal_mask, mem_attn_mask)
        if self._pre_norm:
            out = self.ln_final(out)
        return out

    @property
    def state_batch_axis(self):
        ret = []
        for i in range(self.num_layers):
            if self.recurrent:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            ret.append(layer.state_batch_axis)
        return ret

    def init_states(self, batch_size, ctx, dtype):
        """Initialize the states required for incremental decoding

        Returns
        -------
        init_key :
            Shape (batch_size, 0, N, C_key)
        init_value :
            Shape (batch_size, 0, N, C_value)
        """
        states = []
        for i in range(self.num_layers):
            if self.recurrent:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            states.append(layer.init_states(batch_size=batch_size,
                                            ctx=ctx,
                                            dtype=dtype))
        return states

    def incremental_decode(self, F, data, states, mem, mem_valid_length):
        """Incrementally generate the output given the decoder input.

        Parameters
        ----------
        F
        data
            Shape (batch_size, 1, C_in)
        states
            The previous states, contain a list of
            - prev_multi_key
                Shape (batch_size, prev_seq_length, num_heads, C_key)
            - prev_multi_value
                Shape (batch_size, prev_seq_length, num_heads, C_value)
        mem
            The memory
            Shape (batch_size, mem_length, C_mem)
        mem_valid_length
            Valid length of the memory
            Shape (batch_size,)

        Returns
        -------
        out
            Shape (batch_size, 1, C_out)
        new_states
            The updated states, contain a list of
            - new_key
                Shape (batch_size, prev_seq_length + 1, num_heads, C_key)
            - new_value
                Shape (batch_size, prev_seq_length + 1, num_heads, C_value)
        """
        # 1. Embed the data
        out = self.dropout_layer(data)
        if self._data_norm:
            out = self.ln_data(out)
        mem_attn_mask = gen_mem_attn_mask(F, mem, mem_valid_length, data, None,
                                          dtype=self._dtype)
        new_states = []
        for i in range(self.num_layers):
            if self.recurrent:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            out, new_state = layer.incremental_decode(F, out, states[i],
                                                      mem, mem_valid_length, mem_attn_mask)
            new_states.append(new_state)
        if self._pre_norm:
            out = self.ln_final(out)
        return out, new_states


@use_np
class TransformerNMTModel(HybridBlock):
    def __init__(self, src_vocab_size: int,
                 tgt_vocab_size: int,
                 max_src_length: Optional[int] = None,
                 max_tgt_length: Optional[int] = None,
                 scale_embed: bool = True,
                 pos_embed_type="sinusoidal",
                 shared_embed: bool = True,
                 tie_weights: bool = True,
                 activation_dropout: float = 0.0,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 layer_norm_eps: float = 1E-5,
                 data_norm: bool = False,
                 enc_units: int = 512,
                 enc_hidden_size: int = 2048,
                 enc_num_heads: int = 8,
                 enc_num_layers: int = 6,
                 enc_recurrent: bool = False,
                 enc_activation='relu',
                 enc_pre_norm: bool = False,
                 dec_units: int = 512,
                 dec_hidden_size: int = 2048,
                 dec_num_heads: int = 8,
                 dec_num_layers: int = 6,
                 dec_recurrent: bool = False,
                 dec_activation='relu',
                 dec_pre_norm: bool = False,
                 embed_initializer=mx.init.Xavier('gaussian', 'in', 1),
                 weight_initializer=mx.init.Xavier('uniform', 'avg', 3),
                 bias_initializer='zeros',
                 dtype='float32',
                 prefix=None, params=None):
        """

        Parameters
        ----------
        src_vocab_size
            The vocabulary size of the source language
        tgt_vocab_size
            The vocabulary size of the target language
        max_src_length
            The maximal length of the source sequence.
            If it's negative, we will use treat it as not set.
        max_tgt_length
            The maximal length of the target sequence.
            If it's negative, we will use treat it as not set.
        scale_embed
            Whether to multiply the src and dst embeddings by sqrt(units)
        pos_embed_type
            Type of the positional embedding
        shared_embed
            Whether to share the embedding of the src and tgt language
        tie_weights
            Whether to tie the weights of input + output.
        activation_dropout
            The ratio of the activation dropout in FFN
        dropout
            The default dropout ratio
        attention_dropout
            The ratio of the attention dropout
        layer_norm_eps
            The epsilon of the layer normalization
        data_norm
            Whether to add layer normalization layer after the input.
        enc_units
            Units of the encoder
        enc_hidden_size
            Hidden size of the encoder
        enc_num_heads
            Number of heads of the encoder
        enc_num_layers
            Number of layers of the encoder
        enc_recurrent
            Whether to use recurrent encoder (share weights)
        enc_activation
            Activation of the encoder layer
        enc_pre_norm
            Whether to add layer_norm before self-attention in the encoder
        dec_units
            Units of the decoder
        dec_hidden_size
            Hidden size of the decoder
        dec_num_heads
            Number of heads of the decoder
        dec_num_layers
            Number of layers of the decoder
        dec_recurrent
            Whether to use recurrent decoder (share weights)
        dec_activation
            Activation of the decoder layer
        dec_pre_norm
            Whether to add layer_norm before self-attention in the decoder
        embed_initializer
            Initializer of the embedding layer
        weight_initializer
            Initializer of the weight
        bias_initializer
            Initializer of the bias
        dtype
            Data type of the weights
        prefix
        params
        """
        super(TransformerNMTModel, self).__init__(prefix=prefix, params=params)
        assert src_vocab_size > 0 and tgt_vocab_size > 0,\
            'Cannot set "src_vocab_size" and "tgt_vocab_size" to negative numbers. ' \
            'Are you creating ' \
            'the model with the config from TransformerNMTModel.get_cfg()? If that is ' \
            'the case, you will need to set the cfg.MODEL.src_vocab_size and ' \
            'cfg.MODEL.tgt_vocab_size manually before passing to ' \
            'TransformerNMTModel.from_cfg().'
        self._dtype = dtype
        self._src_vocab_size = src_vocab_size
        self._tgt_vocab_size = tgt_vocab_size
        self.pos_embed_type = pos_embed_type
        self.scaled_embed = scale_embed
        self.enc_units = enc_units
        self.dec_units = dec_units
        if max_src_length is not None and max_src_length < 0:
            max_src_length = None
        if max_tgt_length is not None and max_tgt_length < 0:
            max_tgt_length = None
        if enc_units != dec_units:
            assert shared_embed is False, 'Cannot share embedding when the enc_units and dec_units ' \
                                          'are different! enc_units={},' \
                                          ' dec_units={}'.format(enc_units, dec_units)
        with self.name_scope():
            self.src_embed_layer = nn.Embedding(input_dim=src_vocab_size,
                                                output_dim=enc_units,
                                                prefix='src_embed_',
                                                weight_initializer=embed_initializer,
                                                dtype=self._dtype)
            self.tgt_embed_layer = nn.Embedding(input_dim=tgt_vocab_size,
                                                output_dim=dec_units,
                                                prefix='tgt_embed_',
                                                params=self.src_embed_layer.params
                                                if shared_embed else None,
                                                weight_initializer=embed_initializer,
                                                dtype=self._dtype)
            if pos_embed_type is not None:
                self.src_pos_embed_layer = PositionalEmbedding(units=enc_units,
                                                               max_length=max_src_length,
                                                               dtype=self._dtype,
                                                               method=pos_embed_type,
                                                               prefix='src_pos_embed_')
                self.tgt_pos_embed_layer = PositionalEmbedding(units=dec_units,
                                                               max_length=max_tgt_length,
                                                               dtype=self._dtype,
                                                               method=pos_embed_type,
                                                               prefix='tgt_pos_embed_')
            self.encoder = TransformerEncoder(num_layers=enc_num_layers,
                                              recurrent=enc_recurrent,
                                              units=enc_units,
                                              hidden_size=enc_hidden_size,
                                              num_heads=enc_num_heads,
                                              activation_dropout=activation_dropout,
                                              dropout=dropout,
                                              attention_dropout=attention_dropout,
                                              layer_norm_eps=layer_norm_eps,
                                              weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer,
                                              activation=enc_activation,
                                              data_norm=data_norm,
                                              pre_norm=enc_pre_norm,
                                              dtype=self._dtype,
                                              prefix='enc_')
            self.decoder = TransformerDecoder(num_layers=dec_num_layers,
                                              recurrent=dec_recurrent,
                                              units=dec_units,
                                              mem_units=enc_units,
                                              hidden_size=dec_hidden_size,
                                              num_heads=dec_num_heads,
                                              activation_dropout=activation_dropout,
                                              dropout=dropout,
                                              attention_dropout=attention_dropout,
                                              layer_norm_eps=layer_norm_eps,
                                              weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer,
                                              activation=dec_activation,
                                              data_norm=data_norm,
                                              pre_norm=dec_pre_norm,
                                              dtype=self._dtype,
                                              prefix='dec_')
            if tie_weights:
                self.tgt_final_layer =\
                    nn.Dense(tgt_vocab_size, flatten=False,
                             bias_initializer=bias_initializer,
                             use_bias=False,
                             dtype=self._dtype,
                             params=self.tgt_embed_layer.collect_params(),
                             prefix='tgt_final_')
            else:
                self.tgt_final_layer = \
                    nn.Dense(tgt_vocab_size,
                             flatten=False,
                             weight_initializer=weight_initializer,
                             bias_initializer=bias_initializer,
                             use_bias=False,
                             dtype=self._dtype,
                             prefix='tgt_final_')
        self.encoder.hybridize()
        self.decoder.hybridize()

    @property
    def src_vocab_size(self):
        return self._src_vocab_size

    @property
    def tgt_vocab_size(self):
        return self._tgt_vocab_size

    # TODO(sxjscience) We can actually try to hybridize this function via the
    #  newly-introduced deferred compute.
    def encode(self, F, src_data, src_valid_length):
        """Encode the source data to memory

        Parameters
        ----------
        F
        src_data :
            Shape (batch_size, src_length)
        src_valid_length :
            Shape (batch_size,)

        Returns
        -------
        enc_out :
            Shape (batch_size, src_length, C_out)
        """
        src_data = self.src_embed_layer(src_data)
        if self.scaled_embed:
            src_data = src_data * np.sqrt(self.enc_units)
        if self.pos_embed_type is not None:
            src_data = src_data + self.src_pos_embed_layer(F.npx.arange_like(src_data, axis=1))
        enc_out = self.encoder(src_data, src_valid_length)
        return enc_out

    def decode_seq(self, F, tgt_data, tgt_valid_length, mem_data, mem_valid_length):
        """Decode a sequence of inputs

        Parameters
        ----------
        F
        tgt_data :
            Shape (batch_size, tgt_length)
        tgt_valid_length :
            Shape (batch_size,)
        mem_data :
            Shape (batch_size, src_length, C_out)
        mem_valid_length :
            Shape (batch_size,)

        Returns
        -------
        dec_out :
            Shape (batch_size, tgt_length, tgt_vocab_size)
        """
        tgt_data = self.tgt_embed_layer(tgt_data)
        if self.scaled_embed:
            tgt_data = tgt_data * np.sqrt(self.dec_units)
        if self.pos_embed_type is not None:
            tgt_data = tgt_data + self.tgt_pos_embed_layer(
                F.npx.arange_like(tgt_data, axis=1))
        dec_out = self.decoder(tgt_data, tgt_valid_length, mem_data, mem_valid_length)
        dec_out = self.tgt_final_layer(dec_out)
        return dec_out

    def hybrid_forward(self, F, src_data, src_valid_length, tgt_data, tgt_valid_length):
        """

        Parameters
        ----------
        F
        src_data :
            Shape (batch_size, src_length)
        src_valid_length :
            Shape (batch_size,)
        tgt_data :
            Shape (batch_size, tgt_length)
        tgt_valid_length :
            Shape (batch_size,)

        Returns
        -------
        out :
            Shape (batch_size, tgt_length, tgt_vocab_size)
        """
        enc_out = self.encode(F, src_data, src_valid_length)
        dec_out = self.decode_seq(F, tgt_data, tgt_valid_length, enc_out, src_valid_length)
        return dec_out

    @classmethod
    def get_cfg(cls, key=None):
        if key is None:
            # Use Transformer WMT EN-DE Base
            return transformer_nmt_base()
        else:
            return transformer_nmt_cfg_reg.create(key)

    @classmethod
    def from_cfg(cls, cfg, prefix=None, params=None):
        cfg = cls.get_cfg().clone_merge(cfg)
        embed_initializer = mx.init.create(*cfg.INITIALIZER.embed)
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
        return cls(src_vocab_size=cfg.MODEL.src_vocab_size,
                   tgt_vocab_size=cfg.MODEL.tgt_vocab_size,
                   max_src_length=cfg.MODEL.max_src_length,
                   max_tgt_length=cfg.MODEL.max_tgt_length,
                   scale_embed=cfg.MODEL.scale_embed,
                   pos_embed_type=cfg.MODEL.pos_embed_type,
                   shared_embed=cfg.MODEL.shared_embed,
                   tie_weights=cfg.MODEL.tie_weights,
                   attention_dropout=cfg.MODEL.attention_dropout,
                   activation_dropout=cfg.MODEL.activation_dropout,
                   dropout=cfg.MODEL.dropout,
                   enc_num_layers=cfg.MODEL.ENCODER.num_layers,
                   enc_units=cfg.MODEL.ENCODER.units,
                   enc_num_heads=cfg.MODEL.ENCODER.num_heads,
                   enc_hidden_size=cfg.MODEL.ENCODER.hidden_size,
                   enc_recurrent=cfg.MODEL.ENCODER.recurrent,
                   enc_activation=cfg.MODEL.ENCODER.activation,
                   enc_pre_norm=cfg.MODEL.ENCODER.pre_norm,
                   dec_num_layers=cfg.MODEL.DECODER.num_layers,
                   dec_units=cfg.MODEL.DECODER.units,
                   dec_num_heads=cfg.MODEL.DECODER.num_heads,
                   dec_hidden_size=cfg.MODEL.DECODER.hidden_size,
                   dec_recurrent=cfg.MODEL.DECODER.recurrent,
                   dec_activation=cfg.MODEL.DECODER.activation,
                   dec_pre_norm=cfg.MODEL.DECODER.pre_norm,
                   embed_initializer=embed_initializer,
                   weight_initializer=weight_initializer,
                   bias_initializer=bias_initializer,
                   prefix=prefix,
                   params=params)


@use_np
class TransformerNMTInference(HybridBlock, BaseStepDecoder):
    def __init__(self, model, prefix=None, params=None):
        """

        Parameters
        ----------
        model
        prefix
        params
        """
        super(TransformerNMTInference, self).__init__(prefix=prefix, params=params)
        self.model = model

    def initialize(self, **kwargs):
        # Manually disable the initialize
        raise NotImplementedError('You can not initialize a TransformerNMTFastInference Model! '
                                  'The correct approach is to create a TransformerNMTModel and '
                                  'then build the TransformerNMTInference with the given model.')

    @property
    # TODO(sxjscience) Think about how to improve this
    def state_batch_axis(self) -> Tuple[int, int, int, List]:
        """Return a data structure that stores the batch axis of the internal states
         of the inference model.

        Returns
        -------
        enc_out_batch_axis : int
        src_valid_length_batch_axis : int
        position_batch_axis : int
        dec_layer_batch_axis : list
        """
        return 0, 0, 0, self.model.decoder.state_batch_axis

    def init_states(self, src_data, src_valid_length):  # TODO(sxjscience) Revisit here, support auxiliary states?
        """Initialize the states required for sequence sampling

        Parameters
        ----------
        src_data :
            Shape (batch_size, src_length)
        src_valid_length :
            Shape (batch_size,)

        Returns
        -------
        enc_out :
            Shape (batch_size, src_length, C_mem)
        src_valid_length :
            Shape (batch_size,)
        position :
            Shape (batch_size,)
        dec_states: list
            The states of the decoder
        """
        batch_size = src_data.shape[0]
        ctx = src_data.ctx
        enc_out = self.model.encode(mx.nd, src_data, src_valid_length)
        position = mx.np.zeros((batch_size, 1), dtype=np.int32, ctx=ctx)
        dtype = enc_out.dtype
        dec_states = self.model.decoder.init_states(batch_size, ctx, dtype)
        return enc_out, src_valid_length, position, dec_states

    def hybrid_forward(self, F, step_data, states):
        """

        Parameters
        ----------
        step_data :
            Shape (batch_size,)
        states : tuple
            It includes :
                mem_data : (batch_size, src_length, C_mem)
                mem_valid_length : (batch_size,)
                position : (batch_size,)
                dec_states : list
        Returns
        -------
        out :
            Shape (batch_size, C)
        new_states : tuple
            Has the same structure as the states
        """
        mem_data, mem_valid_length, position, dec_states = states
        # 1. Get the embedding
        step_data = F.np.expand_dims(step_data, axis=1)
        step_data = self.model.tgt_embed_layer(step_data)
        if self.model.scaled_embed:
            step_data = step_data * np.sqrt(self.model.dec_units)
        if self.model.pos_embed_type is not None:
            step_data = step_data + self.model.tgt_pos_embed_layer(position)
        out, new_states =\
            self.model.decoder.incremental_decode(F, step_data, dec_states,
                                                  mem_data, mem_valid_length)
        out = self.model.tgt_final_layer(out)
        out = F.npx.reshape(out, (-2, -1))
        return out, (mem_data, mem_valid_length, position + 1, new_states)
