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
Bert Model

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

__all__ = ['BertModel', 'BertForMLM', 'BertForPretrain',
           'list_pretrained_bert', 'get_pretrained_bert']

import os
from typing import Tuple

import mxnet as mx
from mxnet import use_np
from mxnet.gluon import HybridBlock, nn
from ..registry import BACKBONE_REGISTRY
from .transformer import TransformerEncoderLayer
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from ..utils.config import CfgNode as CN
from ..utils.misc import load_checksum_stats, download
from ..initializer import TruncNorm
from ..attention_cell import MultiHeadAttentionCell, gen_self_attn_mask
from ..layers import get_activation, PositionalEmbedding, PositionwiseFFN, InitializerType
from ..op import select_vectors_by_position
from ..data.tokenizers import HuggingFaceWordPieceTokenizer


PRETRAINED_URL = {
    'google_en_cased_bert_base': {
        'cfg': 'google_en_cased_bert_base/model-5620839a.yml',
        'vocab': 'google_en_cased_bert_base/vocab-c1defaaa.json',
        'params': 'google_en_cased_bert_base/model-c566c289.params',
        'mlm_params': 'google_en_cased_bert_base/model_mlm-c3ff36a3.params',
    },

    'google_en_uncased_bert_base': {
        'cfg': 'google_en_uncased_bert_base/model-4d8422ad.yml',
        'vocab': 'google_en_uncased_bert_base/vocab-e6d2b21d.json',
        'params': 'google_en_uncased_bert_base/model-3712e50a.params',
        'mlm_params': 'google_en_uncased_bert_base/model_mlm-2a23a633.params',
        'lowercase': True,
    },
    'google_en_cased_bert_large': {
        'cfg': 'google_en_cased_bert_large/model-9e127fee.yml',
        'vocab': 'google_en_cased_bert_large/vocab-c1defaaa.json',
        'params': 'google_en_cased_bert_large/model-7aa93704.params',
        'mlm_params': 'google_en_cased_bert_large/model_mlm-d6443fe9.params',
    },
    'google_en_uncased_bert_large': {
        'cfg': 'google_en_uncased_bert_large/model-d0c37dcc.yml',
        'vocab': 'google_en_uncased_bert_large/vocab-e6d2b21d.json',
        'params': 'google_en_uncased_bert_large/model-e53bbc57.params',
        'mlm_params': 'google_en_uncased_bert_large/model_mlm-f5cb8678.params',
        'lowercase': True,
    },
    'google_zh_bert_base': {
        'cfg': 'google_zh_bert_base/model-9b16bda6.yml',
        'vocab': 'google_zh_bert_base/vocab-711c13e4.json',
        'params': 'google_zh_bert_base/model-2efbff63.params',
        'mlm_params': 'google_zh_bert_base/model_mlm-75339658.params',
    },
    'google_multi_cased_bert_base': {
        'cfg': 'google_multi_cased_bert_base/model-881ad607.yml',
        'vocab': 'google_multi_cased_bert_base/vocab-016e1169.json',
        'params': 'google_multi_cased_bert_base/model-c2110078.params',
        'mlm_params': 'google_multi_cased_bert_base/model_mlm-4611e7a3.params',
    },
    'google_en_cased_bert_wwm_large': {
        'cfg': 'google_en_cased_bert_wwm_large/model-9e127fee.yml',
        'vocab': 'google_en_cased_bert_wwm_large/vocab-c1defaaa.json',
        'params': 'google_en_cased_bert_wwm_large/model-0fe841cf.params',
        'mlm_params': None,
    },
    'google_en_uncased_bert_wwm_large': {
        'cfg': 'google_en_uncased_bert_wwm_large/model-d0c37dcc.yml',
        'vocab': 'google_en_uncased_bert_wwm_large/vocab-e6d2b21d.json',
        'params': 'google_en_uncased_bert_wwm_large/model-cb3ad3c2.params',
        'mlm_params': None,
        'lowercase': True,
    }
}


FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'bert.txt'))


@use_np
class BertTransformer(HybridBlock):
    def __init__(self, units: int = 512,
                 hidden_size: int = 2048,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 attention_dropout_prob: float = 0.,
                 hidden_dropout_prob: float = 0.,
                 output_attention: bool = False,
                 dtype='float32',
                 output_all_encodings: bool = False,
                 layer_norm_eps: float = 1E-12,
                 weight_initializer: InitializerType = TruncNorm(stdev=0.02),
                 bias_initializer: InitializerType = 'zeros',
                 activation='gelu',
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        assert units % num_heads == 0,\
            'In BertTransformer, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)

        self._dtype = dtype
        self._num_layers = num_layers
        self._output_attention = output_attention
        self._output_all_encodings = output_all_encodings

        with self.name_scope():
            self.all_layers = nn.HybridSequential(prefix='layers_')
            with self.all_layers.name_scope():
                for layer_idx in range(num_layers):
                    self.all_layers.add(
                      TransformerEncoderLayer(units=units,
                                              hidden_size=hidden_size,
                                              num_heads=num_heads,
                                              attention_dropout_prob=attention_dropout_prob,
                                              hidden_dropout_prob=hidden_dropout_prob,
                                              layer_norm_eps=layer_norm_eps,
                                              weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer,
                                              activation=activation,
                                              prefix='{}_'.format(layer_idx)))

    def hybrid_forward(self, F, data, valid_length):
        """
        Generate the representation given the inputs.

        This is used in training or fine-tuning a bert model.

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
class BertModel(HybridBlock):
    def __init__(self,
                 vocab_size=30000,
                 units=768,
                 hidden_size=3072,
                 num_layers=12,
                 num_heads=12,
                 max_length=512,
                 hidden_dropout_prob=0.,
                 attention_dropout_prob=0.,
                 num_token_types=2,
                 pos_embed_type='learned',
                 activation='gelu',
                 layer_norm_eps=1E-12,
                 embed_initializer=TruncNorm(stdev=0.02),
                 weight_initializer=TruncNorm(stdev=0.02),
                 bias_initializer='zeros',
                 dtype='float32',
                 use_pooler=True,
                 prefix=None,
                 params=None):
        super().__init__(prefix=prefix, params=params)
        self._dtype = dtype
        self.use_pooler = use_pooler
        self.pos_embed_type = pos_embed_type
        self.num_token_types = num_token_types
        self.vocab_size = vocab_size
        self.units = units
        self.max_length = max_length
        self.activation = activation
        self.embed_initializer = embed_initializer
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.layer_norm_eps = layer_norm_eps
        with self.name_scope():
            # Construct BertTransformer
            self.encoder = BertTransformer(
                units=units,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                attention_dropout_prob=attention_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                output_attention=False,
                output_all_encodings=False,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
                dtype=dtype,
                prefix='enc_',
            )
            self.encoder.hybridize()
            # Construct word embedding
            self.word_embed = nn.Embedding(input_dim=vocab_size,
                                           output_dim=units,
                                           weight_initializer=embed_initializer,
                                           dtype=dtype,
                                           prefix='word_embed_')
            self.embed_layer_norm = nn.LayerNorm(epsilon=self.layer_norm_eps,
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
            if self.use_pooler:
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

        This is used in training or fine-tuning a bert model.

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
        initial_embedding = self.get_initial_embedding(F, inputs, token_types)
        prev_out = initial_embedding
        outputs = []

        contextual_embeddings, additional_outputs = self.encoder(prev_out, valid_length)
        outputs.append(contextual_embeddings)
        if self.use_pooler:
            pooled_out = self.apply_pooling(contextual_embeddings)
            outputs.append(pooled_out)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def get_initial_embedding(self, F, inputs, token_types=None):
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
        embedding = self.word_embed(inputs)
        if token_types is None:
            token_types = F.np.zeros_like(inputs)
        type_embedding = self.token_type_embed(token_types)
        embedding = embedding + type_embedding
        if self.pos_embed_type is not None:
            positional_embedding = self.token_pos_embed(F.npx.arange_like(inputs, axis=1))
            positional_embedding = F.np.expand_dims(positional_embedding, axis=0)
            embedding = embedding + positional_embedding
        # Extra layer normalization plus dropout
        embedding = self.embed_layer_norm(embedding)
        embedding = self.embed_dropout(embedding)
        return embedding

    def apply_pooling(self, sequence):
        """Generate the representation given the inputs.

        This is used for pre-training or fine-tuning a bert model.
        Get the first token of the whole sequence which is [CLS]

        sequence:
            Shape (batch_size, sequence_length, units)
        return:
            Shape (batch_size, units)
        """
        outputs = sequence[:, 0, :]
        return self.pooler(outputs)

    @staticmethod
    def get_cfg(key=None):
        if key is None:
            cfg = CN()
            # Parameters for thr small model
            cfg.MODEL = CN()
            cfg.MODEL.vocab_size = 30000
            cfg.MODEL.units = 256
            cfg.MODEL.hidden_size = 1024
            cfg.MODEL.max_length = 512
            cfg.MODEL.num_heads = 4
            cfg.MODEL.num_layers = 12
            cfg.MODEL.pos_embed_type = 'learned'
            cfg.MODEL.activation = 'gelu'
            cfg.MODEL.layer_norm_eps = 1E-12
            cfg.MODEL.num_token_types = 2
            cfg.MODEL.hidden_dropout_prob = 0.1
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
    def from_cfg(cls, cfg, use_pooler=True, prefix=None, params=None):
        cfg = BertModel.get_cfg().clone_merge(cfg)
        assert cfg.VERSION == 1, 'Wrong version!'
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
                   num_token_types=cfg.MODEL.num_token_types,
                   pos_embed_type=cfg.MODEL.pos_embed_type,
                   activation=cfg.MODEL.activation,
                   layer_norm_eps=cfg.MODEL.layer_norm_eps,
                   dtype=cfg.MODEL.dtype,
                   embed_initializer=embed_initializer,
                   weight_initializer=weight_initializer,
                   bias_initializer=bias_initializer,
                   use_pooler=use_pooler,
                   prefix=prefix,
                   params=params)


@use_np
class BertForMLM(HybridBlock):
    def __init__(self, backbone_cfg,
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
            self.backbone_model = BertModel.from_cfg(backbone_cfg, prefix='')
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
                self.mlm_decoder.add(nn.LayerNorm(epsilon=self.backbone_model.layer_norm_eps,
                                                  prefix='ln_'))
                # only load the dense weights with a re-initialized bias
                # parameters are stored in 'word_embed_bias' which is
                # not used in original embedding
                self.mlm_decoder.add(nn.Dense(units=self.backbone_model.vocab_size,
                                              flatten=False,
                                              params=self.backbone_model.word_embed.collect_params('.*weight'),
                                              bias_initializer=bias_initializer,
                                              prefix='score_'))
            self.mlm_decoder.hybridize()

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
        mlm_scores :
            Shape (batch_size, num_masked_positions, vocab_size)
        """
        contextual_embeddings, pooled_out = self.backbone_model(inputs, token_types, valid_length)
        mlm_features = select_vectors_by_position(F, contextual_embeddings, masked_positions)
        mlm_scores = self.mlm_decoder(mlm_features)
        return contextual_embeddings, pooled_out, mlm_scores


@use_np
class BertForPretrain(HybridBlock):
    def __init__(self, backbone_cfg,
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
            self.backbone_model = BertModel.from_cfg(backbone_cfg, prefix='')
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
                self.mlm_decoder.add(nn.LayerNorm(epsilon=self.backbone_model.layer_norm_eps,
                                                  prefix='ln_'))
                # only load the dense weights with a re-initialized bias
                # parameters are stored in 'word_embed_bias' which is
                # not used in original embedding
                self.mlm_decoder.add(nn.Dense(units=self.backbone_model.vocab_size,
                                              flatten=False,
                                              params=self.backbone_model.word_embed.collect_params('.*weight'),
                                              bias_initializer=bias_initializer,
                                              prefix='score_'))
            self.mlm_decoder.hybridize()

    def hybrid_forward(self, F, inputs, token_types, valid_length,
                       masked_positions):
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a bert model.

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
        mlm_scores = self.mlm_decoder(mlm_features)
        return contextual_embeddings, pooled_out, nsp_score, mlm_scores


def list_pretrained_bert():
    return sorted(list(PRETRAINED_URL.keys()))


def get_pretrained_bert(model_name: str = 'google_en_cased_bert_base',
                        root: str = get_model_zoo_home_dir(),
                        load_backbone=True, load_mlm=False)\
        -> Tuple[CN, HuggingFaceWordPieceTokenizer, str, str]:
    """Get the pretrained bert weights

    Parameters
    ----------
    model_name
        The name of the bert model.
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
        model_name, list_pretrained_bert())
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
    cfg = BertModel.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_params_path, local_mlm_params_path


BACKBONE_REGISTRY.register('bert', [BertModel,
                                    get_pretrained_bert,
                                    list_pretrained_bert])
