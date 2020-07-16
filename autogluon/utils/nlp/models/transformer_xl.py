import numpy as np
import mxnet as mx
from mxnet import use_np
from mxnet.gluon import nn, Block, HybridBlock
from ..attention_cell import multi_head_dot_attn, gen_self_attn_mask, gen_mem_attn_mask,\
    RelAttentionScoreCell, MultiHeadAttentionCell
from ..layers import get_activation, PositionalEmbedding, PositionwiseFFN,\
    AdaptiveEmbedding, ProjectedAdaptiveLogSoftmaxWithLoss
from ..utils.config import CfgNode as CN
from ..sequence_sampler import BaseStepDecoder
__all__ = ['TransformerXLDecoderLayer', 'TransformerXLDecoder', 'TransformerXLForLM',
           'TransformerXLForLMGen']


@use_np
class TransformerXLDecoderLayer(HybridBlock):
    def __init__(self, units: int = 512,
                 hidden_size: int = 2048,
                 num_heads: int = 8,
                 activation_dropout: float = 0.0,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 layer_norm_eps: float = 1E-5,
                 activation: str = 'relu',
                 weight_initializer=None,
                 bias_initializer='zeros',
                 pre_norm=False,
                 dtype='float32',
                 layout='NT',
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._pre_norm = pre_norm
        self._dtype = dtype
        self._num_heads = num_heads
        self._layout = layout
        assert layout in ['NT', 'TN'], 'Unknown layout = {}'.format(layout)
        if layout == 'NT':
            self._cell_layout = 'NTK'
        elif layout == 'TN':
            self._cell_layout = 'TNK'
        else:
            raise NotImplementedError
        assert units % num_heads == 0
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.attn_query = nn.Dense(units, in_units=units,
                                       use_bias=False, flatten=False,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       dtype=dtype,
                                       prefix='attn_query_')
            self.attn_kv = nn.Dense(2 * units, in_units=units,
                                    use_bias=False, flatten=False,
                                    weight_initializer=weight_initializer,
                                    bias_initializer=bias_initializer,
                                    dtype=dtype,
                                    prefix='attn_kv_')
            self.rel_pos_score_cell = RelAttentionScoreCell(query_units=units,
                                                            num_heads=num_heads,
                                                            bidirectional=False,
                                                            method='transformer_xl',
                                                            dropout=dropout,
                                                            dtype=dtype,
                                                            layout=self._cell_layout,
                                                            prefix='rel_pos_score_cell_')
            self.attn_cell = MultiHeadAttentionCell(query_units=units,
                                                    num_heads=num_heads,
                                                    attention_dropout=attention_dropout,
                                                    dtype=dtype,
                                                    layout=self._cell_layout,
                                                    prefix='attn_cell_')
            self.out_proj = nn.Dense(units, in_units=units,
                                     use_bias=False, flatten=False,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer,
                                     dtype=dtype,
                                     prefix='out_proj_')
            self.layer_norm = nn.LayerNorm(epsilon=layer_norm_eps,
                                           in_channels=units,
                                           prefix='ln_')
            self.ffn = PositionwiseFFN(units=units,
                                       hidden_size=hidden_size,
                                       activation=activation,
                                       activation_dropout=activation_dropout,
                                       dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       layer_norm_eps=layer_norm_eps,
                                       pre_norm=pre_norm,
                                       dtype=dtype,
                                       prefix='ffn_')

    def hybrid_forward(self, F, data, mem, rel_positions, mask, query_r_bias, query_k_bias):
        """

        Parameters
        ----------
        F
        data
            The input data.
            layout = 'NT':
                Shape (batch_size, query_length, units)
            layout = 'TN':
                Shape (query_length, batch_size, units)
        mem
            The memory.
            layout = 'NT':
                Shape (batch_size, mem_length, units)
            layout = 'TN':
                Shape (mem_length, batch_size, units)
        rel_positions
            The relative positions between data and [mem, data]
            Shape (query_length, mem_length + query_length).
            A positive value means that query is after the memory, i.e.,
            query_location - mem_location.
        mask
            Mask between the query and the memory + query.
            1--> will be used, 0 --> won't be used
            Shape (batch_size, query_length, mem_length + query_length)
        query_r_bias
            The query bias for calculating the relative scores
            Shape (num_heads, query_head_units)
        query_k_bias
            The key bias for calculating the relative scores.
            Shape (num_heads, query_head_units)

        Returns
        -------
        out
            Shape (batch_size, query_length, units)
        """
        if self._layout == 'NT':
            context = F.np.concatenate([mem, data], axis=1)
        elif self._layout == 'TN':
            context = F.np.concatenate([mem, data], axis=0)
        else:
            raise NotImplementedError
        if self._pre_norm:
            query = self.attn_query(self.layer_norm(data))
            key_value = self.attn_kv(self.layer_norm(context))
            key, value = F.np.split(key_value, 2, axis=-1)
        else:
            query = self.attn_query(data)
            key_value = self.attn_kv(context)
            key, value = F.np.split(key_value, 2, axis=-1)
        query = F.npx.reshape(query, (-2, -2, self._num_heads, -1))
        key = F.npx.reshape(key, (-2, -2, self._num_heads, -1))
        value = F.npx.reshape(value, (-2, -2, self._num_heads, -1))
        # Compute attention
        rel_score = self.rel_pos_score_cell(rel_positions, query + query_r_bias)
        out, _ = self.attn_cell(query + query_k_bias, key, value, mask, rel_score)
        out = self.dropout_layer(out)
        if self._pre_norm:
            out = data + out
        else:
            out = self.layer_norm(data + out)
        out = self.ffn(out)
        return out


@use_np
class TransformerXLDecoder(HybridBlock):
    def __init__(self, num_layers=3,
                 units=512,
                 hidden_size=2048,
                 num_heads=8,
                 activation_dropout=0.1,
                 dropout=0.1,
                 attention_dropout=0.0,
                 layernorm_eps=1E-12,
                 activation='relu',
                 dtype='float32',
                 layout='NT',
                 pre_norm=False,
                 weight_initializer=None,
                 bias_initializer=None,
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.query_k_bias = self.params.get('query_k_bias',
                                                shape=(num_heads, units // num_heads),
                                                init=bias_initializer,
                                                allow_deferred_init=True)
            self.query_r_bias = self.params.get('query_r_bias',
                                                shape=(num_heads, units // num_heads),
                                                init=bias_initializer,
                                                allow_deferred_init=True)
            self.decoder_layers = nn.HybridSequential(prefix='l')
            with self.decoder_layers.name_scope():
                for i in range(num_layers):
                    self.decoder_layers.add(
                        TransformerXLDecoderLayer(units=units,
                                                  hidden_size=hidden_size,
                                                  num_heads=num_heads,
                                                  activation_dropout=activation_dropout,
                                                  dropout=dropout,
                                                  attention_dropout=attention_dropout,
                                                  layer_norm_eps=layernorm_eps,
                                                  activation=activation,
                                                  dtype=dtype,
                                                  layout=layout,
                                                  pre_norm=pre_norm,
                                                  weight_initializer=weight_initializer,
                                                  bias_initializer=bias_initializer,
                                                  prefix='{}_'.format(i)))

    def hybrid_forward(self, F, data, mem_l, rel_positions, mask, **params):
        """

        Parameters
        ----------
        F
        data
            - layout = 'NT':
                Shape (batch_size, query_length)
            - layout = 'TN':
                Shape (query_length, batch_size)
        mem_l
            Contains a list of memory objects, each one will contain:
            - layout = 'NT':
                Shape (batch_size, mem_length, C_i)
            - layout = 'TN':
                Shape (mem_length, batch_size, C_i)
        rel_positions
            The relative positions.
            Shape (query_length, mem_length + query_length)
        mask
            Mask between the query and the memory + query.
            Shape (batch_size, query_length, mem_length + query_length)

        Returns
        -------
        out_l
            Contains a list of hidden states, each will contain:
            - layout = 'NT'
                Shape (batch_size, query_length, C_o)
            - layout = 'TN'
                Shape (query_length, batch_size, C_o)
        """
        query_k_bias = params['query_k_bias']
        query_r_bias = params['query_r_bias']
        out_l = []
        out = data
        for i, layer in enumerate(self.decoder_layers):
            out = layer(out, mem_l[i], rel_positions, mask, query_r_bias, query_k_bias)
            out_l.append(out)
        return out_l


@use_np
class TransformerXLForLM(Block):
    def __init__(self, cfg=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        if cfg is None:
            cfg = TransformerXLForLM.get_cfg()
        else:
            cfg = TransformerXLForLM.get_cfg().clone_merge(cfg)
        self._cfg = cfg
        assert cfg.MODEL.vocab_size > 0
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        embed_initializer = mx.init.create(*cfg.INITIALIZER.embed)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
        self._num_layers = cfg.MODEL.num_layers
        self._layout = cfg.MODEL.layout
        self._units = cfg.MODEL.units
        self._dtype = cfg.MODEL.dtype
        assert cfg.MODEL.units % cfg.MODEL.num_heads == 0
        with self.name_scope():
            self.word_emb = AdaptiveEmbedding(vocab_size=cfg.MODEL.vocab_size,
                                              embed_size=cfg.MODEL.embed_units,
                                              units=cfg.MODEL.units,
                                              cutoffs=cfg.MODEL.cutoffs,
                                              div_val=cfg.MODEL.div_val,
                                              scaled=True,
                                              embedding_initializer=embed_initializer,
                                              weight_initializer=weight_initializer,
                                              dtype=cfg.MODEL.dtype,
                                              prefix='word_emb_')
            self.dropout_layer = nn.Dropout(cfg.MODEL.dropout)
            self.decoder = TransformerXLDecoder(num_layers=cfg.MODEL.num_layers,
                                                units=cfg.MODEL.units,
                                                hidden_size=cfg.MODEL.hidden_size,
                                                num_heads=cfg.MODEL.num_heads,
                                                activation_dropout=cfg.MODEL.activation_dropout,
                                                dropout=cfg.MODEL.dropout,
                                                attention_dropout=cfg.MODEL.attention_dropout,
                                                layernorm_eps=cfg.MODEL.layernorm_eps,
                                                activation=cfg.MODEL.activation,
                                                dtype=cfg.MODEL.dtype,
                                                layout=cfg.MODEL.layout,
                                                pre_norm=cfg.MODEL.pre_norm,
                                                weight_initializer=weight_initializer,
                                                bias_initializer=bias_initializer,
                                                prefix='decoder_')
            if cfg.MODEL.tie_weights and cfg.MODEL.tie_projs:
                crit_params = self.word_emb.collect_params('(.*_embed|.*_inter_proj)')
            elif cfg.MODEL.tie_weights and not cfg.MODEL.tie_projs:
                crit_params = self.word_emb.collect_params('.*_embed')
            elif not cfg.MODEL.tie_weights and cfg.MODEL.tie_projs:
                crit_params = self.word_emb.collect_params('.*_inter_proj')
            else:
                crit_params = None
            self.crit = ProjectedAdaptiveLogSoftmaxWithLoss(
                vocab_size=cfg.MODEL.vocab_size,
                embed_size=cfg.MODEL.embed_units,
                in_units=cfg.MODEL.units,
                cutoffs=cfg.MODEL.cutoffs,
                div_val=cfg.MODEL.div_val,
                dtype=cfg.MODEL.dtype,
                use_bias=True,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
                params=crit_params,
                prefix='crit_')

    @property
    def cfg(self):
        return self._cfg

    @property
    def mem_length(self):
        return self.cfg.MODEL.mem_length

    @staticmethod
    def get_cfg(key=None):
        if key is None:
            config = CN()
            config.MODEL = CN()
            # For embedding
            config.MODEL.vocab_size = 1000
            config.MODEL.cutoffs = []
            config.MODEL.div_val = 1.0
            config.MODEL.embed_units = 512
            config.MODEL.tie_weights = True
            config.MODEL.tie_projs = True
            # For decoder
            config.MODEL.units = 512
            config.MODEL.hidden_size = 2048
            config.MODEL.num_layers = 6
            config.MODEL.num_heads = 8
            config.MODEL.activation = 'relu'
            config.MODEL.pre_norm = False
            config.MODEL.dropout = 0.1
            config.MODEL.activation_dropout = 0.1
            config.MODEL.attention_dropout = 0.0
            config.MODEL.layernorm_eps = 1E-6
            config.MODEL.dtype = 'float32'
            config.MODEL.layout = 'TN'
            # For memory
            config.MODEL.mem_length = 512
            # Initialization
            config.INITIALIZER = CN()
            config.INITIALIZER.weight = ['normal', 0.02]
            config.INITIALIZER.bias = ['zeros']
            config.INITIALIZER.embed = ['normal', 0.02]
        else:
            raise NotImplementedError
        return config

    @classmethod
    def from_cfg(cls, cfg, prefix=None, params=None):
        return cls(cfg=cfg, prefix=prefix, params=params)

    @property
    def state_batch_axis(self):
        if self._layout == 'NT':
            return [0 for _ in range(self._num_layers + 1)]
        elif self._layout == 'TN':
            return [1 for _ in range(self._num_layers + 1)]
        else:
            raise NotImplementedError

    def init_states(self, batch_size, ctx):
        """Initialize the states

        Parameters
        ----------
        batch_size
        ctx
            ctx of the initialized

        Returns
        -------
        mems
            A list of memory states
            - layout = 'NT'
                Shape (B, T, C)
            - layout = 'TN'
                Shape (T, B, C)
        """
        if self._layout == 'NT':
            return [mx.np.zeros((batch_size, 0, self._units), ctx=ctx)
                    for _ in range(self._num_layers)]
        elif self._layout == 'TN':
            return [mx.np.zeros((0, batch_size, self._units), ctx=ctx)
                    for _ in range(self._num_layers)]
        else:
            raise NotImplementedError

    def set_mem_length(self, mem_length: int):
        """

        Parameters
        ----------
        mem_length
            The memory length of the model
        """
        self._cfg.defrost()
        self._cfg.MODEL.mem_length = mem_length
        self._cfg.freeze()

    def forward(self, data, target, mem_l, rel_positions=None, data_mem_mask=None,
                same_length=True, detach_memory=True):
        """

        Parameters
        ----------
        data
            The input data
            - layout == 'NT'
                Shape (B, T)
            - layout == 'TN'
                Shape (T, B)
        target
            The ground truth
            - layout == 'NT'
                Shape (B, T)
            - layout == 'TN'
                Shape (T, B)
        mem_l
            A list of memory objects
            - layout == 'NT'
                Shape (B, T_mem, units)
            - layout == 'TN'
                Shape (T_mem, B, units)
        rel_positions
            Shape (query_length, mem_length + query_length)
            By default, we will use the following relative positions
                       ['I', 'can', 'now', 'use', 'numpy', 'in', 'Gluon@@', 'NLP']
            'in':        5,    4,     3,     2,      1,     0,      -1,      -2
            'Gluon@@':   6,    5,     4,     3,      2,     1,       0,      -1
            'NLP':       7,    6,     5,     4,      3,     2,       1,       0
        data_mem_mask
            Shape (B, query_length, mem_length + query_length)
            Here, 1 --> will be used, 0 --> won't be used.
            By default, we will mask all locations that have distance > mem_length with the
            current token.
            Following is an example in which query_length = 3, mem_length = 4
                        |------- <mem> ----------|--------- <query> ------------|
             <query>   ['I', 'can', 'now', 'use', 'numpy', 'in', 'Gluon@@', 'NLP']
            'numpy':     1,    1,     1,     1,      1,     0,      0,        0
            'in':        0,    1,     1,     1,      1,     1,      0,        0
            'Gluon@@':   0,    0,     1,     1,      1,     1,      1,        0
            'NLP':       0,    0,     0,     1,      1,     1,      1,        1

            Also, we provide the option in which we only mask the future tokens, this is
            supported by setting `causal_only` to True. However, there will be a
            discrepancy between training and inference because the effecitve memory length is
            longer for the later tokens in the query.
                        |------- <mem> ----------|--------- <query> ------------|
             <query>   ['I', 'can', 'now', 'use', 'numpy', 'in', 'Gluon@@', 'NLP']
            'numpy':     1,    1,     1,     1,      1,     0,      0,        0
            'in':        1,    1,     1,     1,      1,     1,      0,        0
            'Gluon@@':   1,    1,     1,     1,      1,     1,      1,        0
            'NLP':       1,    1,     1,     1,      1,     1,      1,        1
        same_length
            Whether to ignore the local masking constraint. See the flag above for more information.
        detach_memory
            Whether to detach the encoded memory from the graph.

        Returns
        -------
        logits
            The selected logits
            - layout == 'NT'
                Shape (B, T)
            - layout == 'TN'
                Shape (T, B)
        new_mem_l
            A list of the updated memory
            - layout == 'NT'
                Each will have shape (B, T, C)
            - layout == 'TN'
                Each will have shape (T, B, C)
        """
        # Note that curr_mem_length will not necessarily be equal to mem_length
        if self._layout == 'NT':
            time_axis = 1
            batch_axis = 0
        elif self._layout == 'TN':
            time_axis = 0
            batch_axis = 1
        else:
            raise NotImplementedError
        query_length = data.shape[time_axis]
        curr_mem_length = mem_l[0].shape[time_axis]
        batch_size = mem_l[0].shape[batch_axis]
        ctx = data.ctx
        local_attn_mask = mx.np.ones((batch_size, query_length, curr_mem_length + query_length),
                                     dtype=np.int32, ctx=ctx)
        if same_length:
            # Generate the mask, we mask out the input outside the local self.mem_length window
            local_attn_mask = mx.np.triu(mx.np.tril(local_attn_mask, curr_mem_length),
                                         curr_mem_length - self.mem_length)
        else:
            local_attn_mask = mx.np.tril(local_attn_mask, curr_mem_length)
        if data_mem_mask is None:
            data_mem_mask = local_attn_mask
        else:
            data_mem_mask = data_mem_mask * local_attn_mask
        if rel_positions is None:
            query_ids = mx.np.arange(curr_mem_length, curr_mem_length + query_length,
                                     dtype=np.int32, ctx=ctx)
            mem_ids = mx.np.arange(0, curr_mem_length + query_length,
                                   dtype=np.int32, ctx=ctx)
            rel_positions = mx.np.expand_dims(query_ids, axis=1)\
                            - mx.np.expand_dims(mem_ids, axis=0)
        # Get word embeddings
        word_embeddings = self.word_emb(data)
        word_embeddings = self.dropout_layer(word_embeddings)
        out_l = self.decoder(word_embeddings, mem_l, rel_positions, data_mem_mask)
        # Get the output logits
        logits = self.crit(out_l[-1], target)

        # Get the new memory
        new_mem_l = []
        for step_out, mem in zip([word_embeddings] + out_l, mem_l):
            new_mem = mx.np.concatenate([mem, step_out], axis=time_axis)
            if self._layout == 'NT':
                new_mem = new_mem[:, -self.mem_length:]
            elif self._layout == 'TN':
                new_mem = new_mem[-self.mem_length:, :]
            else:
                raise NotImplementedError
            if detach_memory:
                new_mem_l.append(new_mem.detach())
            else:
                new_mem_l.append(new_mem)
        return logits, new_mem_l

    def step_forward(self, step_data, mem_l):
        """Forward for just one step

        Parameters
        ----------
        step_data
            Shape (B,)
        mem_l
            A list of memory objects
            - layout == 'NT'
                Shape (B, T_mem, units)
            - layout == 'TN'
                Shape (T_mem, B, units)

        Returns
        -------
        logits
            Shape (B, V)
        new_mem_l
            A list of memory objects
            - layout == 'NT'
                Shape (B, min(T_mem + 1, memory_length), C)
            - layout == 'TN'
                Shape (min(T_mem + 1, memory_length), B, C)
        """
        batch_size = step_data.shape[0]
        if self._layout == 'NT':
            curr_mem_length = mem_l[0].shape[1]
        elif self._layout == 'TN':
            curr_mem_length = mem_l[0].shape[0]
        else:
            raise NotImplementedError
        ctx = step_data.ctx
        mask = mx.np.ones((batch_size, 1, curr_mem_length + 1), dtype=np.int32, ctx=ctx)
        rel_positions = mx.np.expand_dims(mx.np.arange(curr_mem_length, -1, -1, dtype=np.int32,
                                                       ctx=ctx), axis=0)
        # Word embedding shape = (B, C)
        word_embeddings = self.dropout_layer(self.word_emb(step_data))
        if self._layout == 'NT':
            word_embeddings = mx.np.expand_dims(word_embeddings, axis=1)
        elif self._layout == 'TN':
            word_embeddings = mx.np.expand_dims(word_embeddings, axis=0)
        else:
            raise NotImplementedError
        out_l = self.decoder(word_embeddings, mem_l, rel_positions, mask)

        # Get logits
        if self._layout == 'NT':
            final_out = out_l[-1][:, 0]
        elif self._layout == 'TN':
            final_out = out_l[-1][0, :]
        else:
            raise NotImplementedError
        logits = self.crit.get_logits(mx, final_out)

        # Update memory
        new_mem_l = []
        for step_out, mem in zip([word_embeddings] + out_l, mem_l):
            if self._layout == 'NT':
                new_mem = mx.np.concatenate([mem, step_out], axis=1)
                new_mem = new_mem[:, -self.mem_length:]
            elif self._layout == 'TN':
                new_mem = mx.np.concatenate([mem, step_out], axis=0)
                new_mem = new_mem[-self.mem_length:, :]
            else:
                raise NotImplementedError
            new_mem_l.append(new_mem)
        return logits, new_mem_l


@use_np
class TransformerXLForLMGen(BaseStepDecoder):
    def __init__(self, net: TransformerXLForLM):
        self.net = net

    def init_states(self, batch_size, ctx):
        return self.net.init_states(batch_size=batch_size, ctx=ctx)

    @property
    def state_batch_axis(self):
        return self.net.state_batch_axis

    def __call__(self, step_data, mem_l):
        return self.net.step_forward(step_data, mem_l)
