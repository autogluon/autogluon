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
RoBERTa Model

@article{liu2019roberta,
    title = {RoBERTa: A Robustly Optimized BERT Pretraining Approach},
    author = {Yinhan Liu and Myle Ott and Naman Goyal and Jingfei Du and
              Mandar Joshi and Danqi Chen and Omer Levy and Mike Lewis and
              Luke Zettlemoyer and Veselin Stoyanov},
    journal={arXiv preprint arXiv:1907.11692},
    year = {2019},
}
"""

__all__ = ['RobertaModel', 'list_pretrained_roberta', 'get_pretrained_roberta']

from typing import Tuple
import os
import mxnet as mx
from mxnet import use_np
from mxnet.gluon import nn, HybridBlock
from .transformer import TransformerEncoderLayer
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from ..utils.config import CfgNode as CN
from ..utils.registry import Registry
from ..utils.misc import load_checksum_stats, download
from ..initializer import TruncNorm
from ..attention_cell import gen_self_attn_mask
from ..registry import BACKBONE_REGISTRY
from ..layers import PositionalEmbedding, get_activation
from ..data.tokenizers import HuggingFaceByteBPETokenizer

PRETRAINED_URL = {
    'fairseq_roberta_base': {
        'cfg': 'fairseq_roberta_base/model-565d1db7.yml',
        'merges': 'fairseq_roberta_base/gpt2-396d4d8e.merges',
        'vocab': 'fairseq_roberta_base/gpt2-f1335494.vocab',
        'params': 'fairseq_roberta_base/model-98b4532f.params'
    },
    'fairseq_roberta_large': {
        'cfg': 'fairseq_roberta_large/model-6e66dc4a.yml',
        'merges': 'fairseq_roberta_large/gpt2-396d4d8e.merges',
        'vocab': 'fairseq_roberta_large/gpt2-f1335494.vocab',
        'params': 'fairseq_roberta_large/model-e3f578dc.params'
    },
    'fairseq_roberta_large_mnli': {
        'cfg': 'fairseq_roberta_large_mnli/model-6e66dc4a.yml',
        'merges': 'fairseq_roberta_large_mnli/gpt2-396d4d8e.merges',
        'vocab': 'fairseq_roberta_large_mnli/gpt2-f1335494.vocab',
        'params': 'fairseq_roberta_large_mnli/model-5288bb09.params'
    }
}

FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'roberta.txt'))
roberta_cfg_reg = Registry('roberta_cfg')


@roberta_cfg_reg.register()
def roberta_base():
    cfg = CN()
    # Config for the roberta base model
    cfg.MODEL = CN()
    cfg.MODEL.vocab_size = 50265
    cfg.MODEL.units = 768
    cfg.MODEL.hidden_size = 3072
    cfg.MODEL.max_length = 512
    cfg.MODEL.num_heads = 12
    cfg.MODEL.num_layers = 12
    cfg.MODEL.pos_embed_type = 'learned'
    cfg.MODEL.activation = 'gelu'
    cfg.MODEL.pooler_activation = 'tanh'
    cfg.MODEL.layer_norm_eps = 1E-5
    cfg.MODEL.hidden_dropout_prob = 0.1
    cfg.MODEL.attention_dropout_prob = 0.1
    cfg.MODEL.dtype = 'float32'
    cfg.INITIALIZER = CN()
    cfg.INITIALIZER.embed = ['truncnorm', 0, 0.02]
    cfg.INITIALIZER.weight = ['truncnorm', 0, 0.02]
    cfg.INITIALIZER.bias = ['zeros']
    cfg.VERSION = 1
    cfg.freeze()
    return cfg


@roberta_cfg_reg.register()
def roberta_large():
    cfg = roberta_base()
    cfg.defrost()
    cfg.MODEL.units = 1024
    cfg.MODEL.hidden_size = 4096
    cfg.MODEL.num_heads = 16
    cfg.MODEL.num_layers = 24
    cfg.freeze()
    return cfg

@use_np
class RobertaModel(HybridBlock):
    def __init__(self,
                 vocab_size=50265,
                 units=768,
                 hidden_size=3072,
                 num_layers=12,
                 num_heads=12,
                 max_length=512,
                 hidden_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 pos_embed_type='learned',
                 activation='gelu',
                 pooler_activation='tanh',
                 layer_norm_eps=1E-5,
                 embed_initializer=TruncNorm(stdev=0.02),
                 weight_initializer=TruncNorm(stdev=0.02),
                 bias_initializer='zeros',
                 dtype='float32',
                 use_pooler=False,
                 use_mlm=True,
                 untie_weight=False,
                 encoder_normalize_before=True,
                 return_all_hiddens=False,
                 prefix=None,
                 params=None):
        """ 

        Parameters
        ----------
        vocab_size
        units
        hidden_size
        num_layers
        num_heads
        max_length
        hidden_dropout_prob
        attention_dropout_prob
        pos_embed_type
        activation
        pooler_activation
        layer_norm_eps
        embed_initializer
        weight_initializer
        bias_initializer
        dtype
        use_pooler
            Whether to use classification head
        use_mlm        
            Whether to use lm head, if False, forward return hidden states only
        untie_weight
            Whether to untie weights between embeddings and classifiers
        encoder_normalize_before
        return_all_hiddens
        prefix
        params
        """
        super(RobertaModel, self).__init__(prefix=prefix, params=params)
        self.vocab_size = vocab_size
        self.units = units
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.pos_embed_type = pos_embed_type
        self.activation = activation
        self.pooler_activation = pooler_activation
        self.layer_norm_eps = layer_norm_eps
        self.dtype = dtype
        self.use_pooler = use_pooler
        self.use_mlm = use_mlm
        self.untie_weight = untie_weight
        self.encoder_normalize_before = encoder_normalize_before
        self.return_all_hiddens = return_all_hiddens
        with self.name_scope():
            self.tokens_embed = nn.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.units,
                weight_initializer=embed_initializer,
                dtype=self.dtype,
                prefix='tokens_embed_'
            )
            if self.encoder_normalize_before:
                self.embed_ln = nn.LayerNorm(
                    epsilon=self.layer_norm_eps,
                    in_channels=self.units,
                    prefix='embed_ln_'
                )
            else:
                self.embed_ln = None
            self.embed_dropout = nn.Dropout(self.hidden_dropout_prob)
            self.pos_embed = PositionalEmbedding(
                units=self.units,
                max_length=self.max_length,
                dtype=self.dtype,
                method=pos_embed_type,
                prefix='pos_embed_'
            )
            
            self.encoder = RobertaEncoder(
                units=self.units,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                attention_dropout_prob=self.attention_dropout_prob,
                hidden_dropout_prob=self.hidden_dropout_prob,
                layer_norm_eps=self.layer_norm_eps,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
                activation=self.activation,
                dtype=self.dtype,
                return_all_hiddens=self.return_all_hiddens
            )
            self.encoder.hybridize()
            
            if self.use_mlm:
                embed_weight = None if untie_weight else \
                    self.tokens_embed.collect_params('.*weight')
                self.lm_head = RobertaLMHead(
                    self.units,
                    self.vocab_size,
                    self.activation,
                    layer_norm_eps=self.layer_norm_eps,
                    embed_weight=embed_weight,
                    weight_initializer=weight_initializer,
                    bias_initializer=bias_initializer
                )
                self.lm_head.hybridize()
            # TODO support use_pooler

    def hybrid_forward(self, F, tokens, valid_length):
        x = self.tokens_embed(tokens)
        if self.pos_embed_type:
            positional_embedding = self.pos_embed(F.npx.arange_like(x, axis=1))
            positional_embedding = F.np.expand_dims(positional_embedding, axis=0)
            x = x + positional_embedding
        if self.embed_ln:
            x = self.embed_ln(x)
        x = self.embed_dropout(x)
        inner_states = self.encoder(x, valid_length)
        x = inner_states[-1]
        if self.use_mlm:
            x = self.lm_head(x)
        if self.return_all_hiddens:
            return x, inner_states
        else:
            return x

    @staticmethod
    def get_cfg(key=None):
        if key:
            return roberta_cfg_reg.create(key)
        else:
            return roberta_base()

    @classmethod
    def from_cfg(cls,
                 cfg,
                 use_pooler=False,
                 use_mlm=True,
                 untie_weight=False,
                 encoder_normalize_before=True,
                 return_all_hiddens=False,
                 prefix=None,
                 params=None):
        cfg = RobertaModel.get_cfg().clone_merge(cfg)
        embed_initializer = mx.init.create(*cfg.INITIALIZER.embed)
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
        return cls(vocab_size=cfg.MODEL.vocab_size,
                   units=cfg.MODEL.units,
                   hidden_size=cfg.MODEL.hidden_size,
                   num_layers=cfg.MODEL.num_layers,
                   num_heads=cfg.MODEL.num_heads,
                   max_length=cfg.MODEL.max_length,
                   hidden_dropout_prob=cfg.MODEL.hidden_dropout_prob,
                   attention_dropout_prob=cfg.MODEL.attention_dropout_prob,
                   pos_embed_type=cfg.MODEL.pos_embed_type,
                   activation=cfg.MODEL.activation,
                   pooler_activation=cfg.MODEL.pooler_activation,
                   layer_norm_eps=cfg.MODEL.layer_norm_eps,
                   embed_initializer=embed_initializer,
                   weight_initializer=weight_initializer,
                   bias_initializer=bias_initializer,
                   dtype=cfg.MODEL.dtype,
                   use_pooler=use_pooler,
                   use_mlm=use_mlm,
                   untie_weight=untie_weight,
                   encoder_normalize_before=encoder_normalize_before,
                   return_all_hiddens=return_all_hiddens,
                   prefix=prefix,
                   params=params)

@use_np
class RobertaEncoder(HybridBlock):    
    def __init__(self,
                 units=768,
                 hidden_size=3072,
                 num_layers=12,
                 num_heads=12,
                 attention_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 layer_norm_eps=1E-5,
                 weight_initializer=TruncNorm(stdev=0.02),
                 bias_initializer='zeros',
                 activation='gelu',
                 dtype='float32',
                 return_all_hiddens=False,
                 prefix='encoder_',
                 params=None):
        super(RobertaEncoder, self).__init__(prefix=prefix, params=params)
        self.units = units
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation
        self.dtype = dtype
        self.return_all_hiddens = return_all_hiddens
        with self.name_scope():
            self.all_layers = nn.HybridSequential(prefix='layers_')
            with self.all_layers.name_scope():
                for layer_idx in range(self.num_layers):
                    self.all_layers.add(
                        TransformerEncoderLayer(
                            units=self.units,
                            hidden_size=self.hidden_size,
                            num_heads=self.num_heads,
                            attention_dropout_prob=self.attention_dropout_prob,
                            hidden_dropout_prob=self.hidden_dropout_prob,
                            layer_norm_eps=self.layer_norm_eps,
                            weight_initializer=weight_initializer,
                            bias_initializer=bias_initializer,
                            activation=self.activation,
                            dtype=self.dtype,
                            prefix='{}_'.format(layer_idx)
                        )
                    )

    def hybrid_forward(self, F, inputs, valid_length):
        """

        Parameters
        ----------
        F
        inputs
            Shape (batch_size, seq_length)
        valid_length
            Shape (batch_size,)

        Returns
        -------
        out
            Either a list that contains all the hidden states or a list with one element
        """
        atten_mask = gen_self_attn_mask(F, inputs, valid_length,
                                        dtype=self.dtype, attn_type='full')
        inner_states = [inputs]
        for layer_idx in range(self.num_layers):
            layer = self.all_layers[layer_idx]
            inputs, _ = layer(inputs, atten_mask)
            inner_states.append(inputs)
        if not self.return_all_hiddens:
            inner_states = [inputs]
        return inner_states


@use_np
class RobertaLMHead(HybridBlock):
    def __init__(self,
                 embed_dim=768,
                 output_dim=50265,
                 activation_fn='gelu',
                 layer_norm_eps=1E-5,
                 embed_weight=None,
                 weight_initializer=TruncNorm(stdev=0.02),
                 bias_initializer='zeros',
                 prefix='lm_',
                 params=None):
        super(RobertaLMHead, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.dense1 = nn.Dense(in_units=embed_dim,
                                   units=embed_dim,
                                   flatten=False,
                                   weight_initializer=weight_initializer,
                                   bias_initializer=bias_initializer,
                                   prefix='dense1_')
            self.activation_fn = get_activation(activation_fn)
            self.ln = nn.LayerNorm(
                epsilon=layer_norm_eps,
                in_channels=embed_dim,
                prefix='ln_')
            if embed_weight:
                # notice the bias of dense2 here
                # will be *tokens_embed_bias
                self.dense2 = nn.Dense(in_units=embed_dim,
                                       units=output_dim,
                                       flatten=False,
                                       params=embed_weight,
                                       bias_initializer='zeros',
                                       prefix='dense2_')
            else:
                self.dense2 = nn.Dense(in_units=embed_dim,
                                       units=output_dim,
                                       activation=None,
                                       flatten=False,
                                       weight_initializer=weight_initializer,
                                       bias_initializer='zeros',
                                       prefix='dense2_')

    def hybrid_forward(self, F, x):
        x = self.dense1(x)
        x = self.activation_fn(x)
        x = self.ln(x)
        x = self.dense2(x)
        return x


def list_pretrained_roberta():
    return sorted(list(PRETRAINED_URL.keys()))


def get_pretrained_roberta(model_name: str = 'fairseq_roberta_base',
                           root: str = get_model_zoo_home_dir()) \
        -> Tuple[CN, HuggingFaceByteBPETokenizer, str]:
    """Get the pretrained RoBERTa weights

    Parameters
    ----------
    model_name
        The name of the RoBERTa model.
    root
        The downloading root

    Returns
    -------
    cfg
        Network configuration
    tokenizer
        The HuggingFaceByteBPETokenizer
    params_path
        Path to the parameters
    """
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}'.format(
        model_name, list_pretrained_roberta())
    cfg_path = PRETRAINED_URL[model_name]['cfg']
    merges_path = PRETRAINED_URL[model_name]['merges']
    vocab_path = PRETRAINED_URL[model_name]['vocab']
    params_path = PRETRAINED_URL[model_name]['params']
    local_paths = dict()
    for k, path in [('cfg', cfg_path), ('vocab', vocab_path),
                    ('merges', merges_path), ('params', params_path)]:
        local_paths[k] = download(url=get_repo_model_zoo_url() + path,
                                  path=os.path.join(root, path),
                                  sha1_hash=FILE_STATS[path])
    tokenizer = HuggingFaceByteBPETokenizer(local_paths['merges'], local_paths['vocab'])
    cfg = RobertaModel.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_paths['params']


BACKBONE_REGISTRY.register('roberta', [RobertaModel,
                                       get_pretrained_roberta,
                                       list_pretrained_roberta])
