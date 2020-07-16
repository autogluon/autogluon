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
"""Electra Model.

@inproceedings{clark2020electra,
  title = {{ELECTRA}: Pre-training Text Encoders as Discriminators Rather Than Generators},
  author = {Kevin Clark and Minh-Thang Luong and Quoc V. Le and Christopher D. Manning},
  booktitle = {ICLR},
  year = {2020},
  url = {https://openreview.net/pdf?id=r1xMH1BtvB}
}

"""
__all__ = ['ElectraModel', 'ElectraDiscriminator', 'ElectraGenerator',
           'ElectraForPretrain', 'list_pretrained_electra', 'get_pretrained_electra']

import os
from typing import Tuple, Optional

import mxnet as mx
import numpy as np
from mxnet import use_np
from mxnet.gluon import HybridBlock, nn
from ..registry import BACKBONE_REGISTRY
from ..op import gumbel_softmax, select_vectors_by_position, updated_vectors_by_position
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from ..layers import PositionalEmbedding, get_activation
from .transformer import TransformerEncoderLayer
from ..initializer import TruncNorm
from ..utils.config import CfgNode as CN
from ..utils.misc import load_checksum_stats, download
from ..attention_cell import gen_self_attn_mask
from ..data.tokenizers import HuggingFaceWordPieceTokenizer


def get_generator_cfg(model_config):
    """
    Get the generator configuration from the Electra model config.
    The size of generator is usually smaller than discriminator but same in electra small,
    which exists  a conflict between source code and original paper.
    """
    generator_cfg = model_config.clone()
    generator_layers_scale = model_config.MODEL.generator_layers_scale
    generator_units_scale = model_config.MODEL.generator_units_scale
    generator_cfg.defrost()
    # the round function is used to slove int(0.3333*768)!=256 for electra base
    generator_cfg.MODEL.units = round(generator_units_scale * model_config.MODEL.units)
    generator_cfg.MODEL.hidden_size = round(generator_units_scale * model_config.MODEL.hidden_size)
    generator_cfg.MODEL.num_heads = round(generator_units_scale * model_config.MODEL.num_heads)
    generator_cfg.MODEL.num_layers = round(generator_layers_scale * model_config.MODEL.num_layers)
    generator_cfg.freeze()
    return generator_cfg


PRETRAINED_URL = {
    'google_electra_small': {
        'cfg': 'google_electra_small/model-9ffb21c8.yml',
        'vocab': 'google_electra_small/vocab-e6d2b21d.json',
        'params': 'google_electra_small/model-2654c8b4.params',
        'disc_model': 'google_electra_small/disc_model-137714b6.params',
        'gen_model': 'google_electra_small/gen_model-d11fd0b1.params',
    },
    'google_electra_base': {
        'cfg': 'google_electra_base/model-5b35ca0b.yml',
        'vocab': 'google_electra_base/vocab-e6d2b21d.json',
        'params': 'google_electra_base/model-31c235cc.params',
        'disc_model': 'google_electra_base/disc_model-514bd353.params',
        'gen_model': 'google_electra_base/gen_model-665ce594.params',
    },
    'google_electra_large': {
        'cfg': 'google_electra_large/model-31b7dfdd.yml',
        'vocab': 'google_electra_large/vocab-e6d2b21d.json',
        'params': 'google_electra_large/model-9baf9ff5.params',
        'disc_model': 'google_electra_large/disc_model-5b820c02.params',
        'gen_model': 'google_electra_large/gen_model-667121df.params',
    }
}

FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'electra.txt'))


@use_np
class ElectraEncoder(HybridBlock):
    def __init__(self, units=512,
                 hidden_size=2048,
                 num_layers=6,
                 num_heads=8,
                 attention_dropout_prob=0.,
                 hidden_dropout_prob=0.,
                 output_attention=False,
                 dtype='float32',
                 output_all_encodings=False,
                 layer_norm_eps=1E-12,
                 weight_initializer=TruncNorm(stdev=0.02),
                 bias_initializer='zeros',
                 activation='gelu', prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        assert units % num_heads == 0, \
            'In ElectraEncoder, The units should be divisible ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)

        self._dtype = dtype
        self._num_layers = num_layers

        self._output_attention = output_attention
        self._output_all_encodings = output_all_encodings

        with self.name_scope():
            self.all_encoder_layers = nn.HybridSequential(prefix='layers_')
            with self.all_encoder_layers.name_scope():
                for layer_idx in range(num_layers):
                    self.all_encoder_layers.add(
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

        This is used in training or fine-tuning a Electra model.

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
            layer = self.all_encoder_layers[layer_idx]
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
class ElectraModel(HybridBlock):
    """Electra Model

    This is almost the same as bert model with embedding_size adjustable (factorized embedding).
    """

    def __init__(self,
                 vocab_size=30000,
                 units=768,
                 hidden_size=3072,
                 embed_size=128,
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
                 tied_embeddings=False,
                 word_embed_params=None,
                 token_type_embed_params=None,
                 token_pos_embed_params=None,
                 embed_layer_norm_params=None,
                 prefix=None,
                 params=None):
        super().__init__(prefix=prefix, params=params)
        self._dtype = dtype
        self.use_pooler = use_pooler
        self.pos_embed_type = pos_embed_type
        self.num_token_types = num_token_types
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.units = units
        self.max_length = max_length
        self.activation = activation
        self.embed_initializer = embed_initializer
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.layer_norm_eps = layer_norm_eps
        with self.name_scope():
            # Construct ElectraEncoder
            self.encoder = ElectraEncoder(
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
            # Construct model embedding which consists of three parts including word embedding,
            # type embedding and positional embeddings.
            # The hyper-parameters "tied_embeddings" is particularly
            # used for sharing the embeddings between the Electra generator and the
            # Electra discriminator.
            if tied_embeddings:
                assert word_embed_params is not None
                assert token_type_embed_params is not None
                assert token_pos_embed_params is not None
                assert embed_layer_norm_params is not None

            self.word_embed = nn.Embedding(input_dim=vocab_size,
                                           output_dim=embed_size,
                                           weight_initializer=embed_initializer,
                                           dtype=dtype,
                                           params=word_embed_params,
                                           prefix='word_embed_')
            # Construct token type embedding
            self.token_type_embed = nn.Embedding(input_dim=num_token_types,
                                                 output_dim=embed_size,
                                                 weight_initializer=weight_initializer,
                                                 params=token_type_embed_params,
                                                 prefix='token_type_embed_')
            self.token_pos_embed = PositionalEmbedding(units=embed_size,
                                                       max_length=max_length,
                                                       dtype=self._dtype,
                                                       method=pos_embed_type,
                                                       params=token_pos_embed_params,
                                                       prefix='token_pos_embed_')
            self.embed_layer_norm = nn.LayerNorm(epsilon=self.layer_norm_eps,
                                                 params=embed_layer_norm_params,
                                                 prefix='embed_ln_')

            self.embed_dropout = nn.Dropout(hidden_dropout_prob)
            if embed_size != units:
                self.embed_factorized_proj = nn.Dense(units=units,
                                                      flatten=False,
                                                      weight_initializer=weight_initializer,
                                                      bias_initializer=bias_initializer,
                                                      prefix='embed_factorized_proj_')

    def hybrid_forward(self, F, inputs, token_types, valid_length=None):
        # pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a Electra model.

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
        # Projecting the embedding into units
        prev_out = initial_embedding
        if self.embed_size != self.units:
            prev_out = self.embed_factorized_proj(prev_out)
        outputs = []
        contextual_embeddings, additional_outputs = self.encoder(prev_out, valid_length)
        outputs.append(contextual_embeddings)
        if self.use_pooler:
            # Here we just get the first token ([CLS]) without any pooling strategy,
            # which is slightly different between bert model with the pooled_out
            # the attribute name is keeping the same as bert and albert model with defualt
            # use_pooler=True
            pooled_out = contextual_embeddings[:, 0, :]
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

    @staticmethod
    def get_cfg(key=None):
        if key is None:
            cfg = CN()
            # Model Parameters for the electra small
            cfg.MODEL = CN()
            cfg.MODEL.vocab_size = 30522
            cfg.MODEL.embed_size = 128
            cfg.MODEL.units = 256
            cfg.MODEL.hidden_size = 1024
            cfg.MODEL.max_length = 512
            cfg.MODEL.num_heads = 4
            cfg.MODEL.num_layers = 12
            cfg.MODEL.pos_embed_type = 'learned'
            # Unlike BERT and ALBERT, which ues gelu(tanh), the gelu(erf) is used in Electra.
            cfg.MODEL.activation = 'gelu'
            cfg.MODEL.layer_norm_eps = 1E-12
            cfg.MODEL.num_token_types = 2
            cfg.MODEL.hidden_dropout_prob = 0.1
            cfg.MODEL.attention_dropout_prob = 0.1
            cfg.MODEL.dtype = 'float32'
            cfg.MODEL.generator_layers_scale = 1.0
            # multiplier for units, hidden_size, and num_heads
            cfg.MODEL.generator_units_scale = 1.0
            # Hyper-parameters of the Initializers
            cfg.INITIALIZER = CN()
            cfg.INITIALIZER.embed = ['truncnorm', 0, 0.02]
            cfg.INITIALIZER.weight = ['truncnorm', 0, 0.02]  # TruncNorm(0, 0.02)
            cfg.INITIALIZER.bias = ['zeros']
            # Version of the model. This helps ensure backward compatibility.
            # Also, we can not use string here due to https://github.com/rbgirshick/yacs/issues/26
            cfg.VERSION = 1
            cfg.freeze()
        else:
            raise NotImplementedError
        return cfg

    @classmethod
    def from_cfg(cls,
                 cfg,
                 use_pooler=True,
                 tied_embeddings=False,
                 word_embed_params=None,
                 token_type_embed_params=None,
                 token_pos_embed_params=None,
                 embed_layer_norm_params=None,
                 prefix=None,
                 params=None):
        cfg = ElectraModel.get_cfg().clone_merge(cfg)
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
                   tied_embeddings=tied_embeddings,
                   word_embed_params=word_embed_params,
                   token_type_embed_params=token_type_embed_params,
                   token_pos_embed_params=token_pos_embed_params,
                   embed_layer_norm_params=embed_layer_norm_params,
                   prefix=prefix,
                   params=params)


@use_np
class ElectraDiscriminator(HybridBlock):
    """
    It is slightly different from the traditional mask language model which recover the
    masked word (find the matched word in dictionary). The Object of Discriminator in
    Electra is 'replaced token detection' that is a binary classification task to
    predicts every token whether it is an original or a replacement.
    """

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
            self.backbone_model = ElectraModel.from_cfg(backbone_cfg, prefix='')
            if weight_initializer is None:
                weight_initializer = self.backbone_model.weight_initializer
            if bias_initializer is None:
                bias_initializer = self.backbone_model.bias_initializer
            self.rtd_encoder = nn.HybridSequential(prefix='disc_')
            with self.rtd_encoder.name_scope():
                # Extra non-linear layer
                self.rtd_encoder.add(nn.Dense(units=self.backbone_model.units,
                                              flatten=False,
                                              weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer,
                                              prefix='proj_'))
                self.rtd_encoder.add(get_activation(self.backbone_model.activation))
                self.rtd_encoder.add(nn.Dense(units=1,
                                              flatten=False,
                                              weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer,
                                              prefix='predctions_'))
            self.rtd_encoder.hybridize()

    def hybrid_forward(self, F, inputs, token_types, valid_length):
        """Getting the scores of the replaced token detection of the whole sentence
        based on the corrupted tokens produced from a generator.

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
        contextual_embedding
            Shape (batch_size, seq_length, units).
        pooled_out
            Shape (batch_size, units)
        rtd_scores
            Shape (batch_size, seq_length)
        """
        contextual_embeddings, pooled_out = self.backbone_model(inputs, token_types, valid_length)
        rtd_scores = self.rtd_encoder(contextual_embeddings).squeeze(-1)
        return contextual_embeddings, pooled_out, rtd_scores


@use_np
class ElectraGenerator(HybridBlock):
    """
    This is a typical mlm model whose size is usually the 1/4 - 1/2 of the discriminator.
    """

    def __init__(self, backbone_cfg,
                 tied_embeddings=True,
                 word_embed_params=None,
                 token_type_embed_params=None,
                 token_pos_embed_params=None,
                 embed_layer_norm_params=None,
                 weight_initializer=None,
                 bias_initializer=None,
                 prefix=None,
                 params=None):
        """

        Parameters
        ----------
        backbone_cfg
            Configuration of the backbone model
        tied_embeddings
            Reuse the embeddings of discriminator
        word_embed_params
            The parameters to load into word embeddings
        token_type_embed_params
            The parameters to load into word token type embeddings
        token_pos_embed_params
            The parameters to load into token positional embeddings
        embed_layer_norm_params
            The parameters to load into token layer normalization layer of embeddings
        weight_initializer
        bias_initializer
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.backbone_model = ElectraModel.from_cfg(
                backbone_cfg,
                tied_embeddings=tied_embeddings,
                word_embed_params=word_embed_params,
                token_type_embed_params=token_type_embed_params,
                token_pos_embed_params=token_pos_embed_params,
                embed_layer_norm_params=embed_layer_norm_params,
                prefix='')
            if weight_initializer is None:
                weight_initializer = self.backbone_model.weight_initializer
            if bias_initializer is None:
                bias_initializer = self.backbone_model.bias_initializer
            self.mlm_decoder = nn.HybridSequential(prefix='gen_')
            with self.mlm_decoder.name_scope():
                # Extra non-linear layer
                self.mlm_decoder.add(nn.Dense(units=self.backbone_model.embed_size,
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
                self.mlm_decoder.add(
                    nn.Dense(
                        units=self.backbone_model.vocab_size,
                        flatten=False,
                        params=self.backbone_model.word_embed.collect_params('.*weight'),
                        bias_initializer=bias_initializer,
                        prefix='score_'))
            self.mlm_decoder.hybridize()

    def hybrid_forward(self, F, inputs, token_types, valid_length, masked_positions):
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
class ElectraForPretrain(HybridBlock):
    """
    A integrated model combined with a generator and a discriminator.  Generator here
    produces a corrupted tokens playing as fake data to fool a discriminator whose
    objective is to distinguish whether each token in the input sentence it accepts
    is the same as the original. It is a classification task instead of prediction
    task as other pretrained models such as bert.
    """

    def __init__(self,
                 disc_cfg,
                 uniform_generator=False,
                 tied_generator=False,
                 tied_embeddings=True,
                 disallow_correct=False,
                 temperature=1.0,
                 dtype='float32',
                 weight_initializer=None,
                 bias_initializer=None,
                 prefix=None,
                 params=None):
        """

        Parameters
        ----------
        disc_cfg :
            Config for discriminator model including scaled size for generator
        uniform_generator :
            Wether to get a generator with uniform weights, the mlm_scores from
            which are totally random. In this case , a discriminator learns from
            a random 15% of the input tokens distinct from the subset.
        tied_generator :
            Whether to tie backbone model weights of generator and discriminator.
            The size of G and D are required to be same if set to True.
        tied_embeddings :
            Whether to tie the embeddings of generator and discriminator
        disallow_correct :
            Whether the correct smaples of generator are allowed,
            that is 15% of tokens are always fake.
        temperature :
            Temperature of gumbel distribution for sampling from generator
        weight_initializer
        bias_initializer
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        self._uniform_generator = uniform_generator
        self._tied_generator = tied_generator
        self._tied_embeddings = tied_embeddings
        self._disallow_correct = disallow_correct
        self._temperature = temperature
        self._dtype = dtype

        self.disc_cfg = disc_cfg
        self.vocab_size = disc_cfg.MODEL.vocab_size
        self.gen_cfg = get_generator_cfg(disc_cfg)
        self.discriminator = ElectraDiscriminator(disc_cfg, prefix='electra_')
        self.disc_backbone = self.discriminator.backbone_model
        if tied_embeddings:
            word_embed_params = self.disc_backbone.word_embed.collect_params()
            token_type_embed_params = self.disc_backbone.token_pos_embed.collect_params()
            token_pos_embed_params = self.disc_backbone.token_pos_embed.collect_params()
            embed_layer_norm_params = self.disc_backbone.embed_layer_norm.collect_params()
        else:
            word_embed_params = None
            token_type_embed_params = None
            token_pos_embed_params = None
            embed_layer_norm_params = None

        if not uniform_generator and not tied_generator:
            self.generator = ElectraGenerator(
                self.gen_cfg,
                tied_embeddings=tied_embeddings,
                word_embed_params=word_embed_params,
                token_type_embed_params=token_type_embed_params,
                token_pos_embed_params=token_pos_embed_params,
                embed_layer_norm_params=embed_layer_norm_params,
                prefix='generator_')
            self.generator.hybridize()

        elif tied_generator:
            # Reuse the weight of the discriminator backbone model
            self.generator = ElectraGenerator(
                self.gen_cfg, tied_embeddings=False, prefix='generator_')
            self.generator.backbone_model = self.disc_backbone
            self.generator.hybridize()
        elif uniform_generator:
            # get the mlm_scores randomly over vocab
            self.generator = None

        self.discriminator.hybridize()

    def hybrid_forward(self, F, inputs, token_types, valid_length,
                       unmasked_tokens, masked_positions):
        """Getting the mlm scores of each masked positions from a generator,
        then produces the corrupted tokens sampling from a gumbel distribution.
        We also get the ground-truth and scores of the replaced token detection
        which is output by a discriminator. The ground-truth is an array with same
        shape as the input using 1 stand for original token and 0 for replacement.

        Notice: There is a problem when the masked positions have duplicate indexs.
        Try to avoid that in the data preprocessing process. In addition, loss calculation
        should be done in the training scripts as well.

        Parameters
        ----------
        F
        inputs :
            The masked input
            Shape (batch_size, seq_length)
        token_types :
            Shape (batch_size, seq_length)

            If the inputs contain two sequences, we will set different token types for the first
             sentence and the second sentence.
        valid_length :
            The valid length of each sequence
            Shape (batch_size,)
        unmasked_tokens :
            The original tokens that appear in the unmasked input sequence
            Shape (batch_size, num_masked_positions).
        masked_positions :
            The masked position of the sequence
            Shape (batch_size, num_masked_positions).

        Returns
        -------
        mlm_scores :
            Shape (batch_size, num_masked_positions, vocab_size)
        rtd_scores :
            Shape (batch_size, seq_length)
        replaced_inputs :
            Shape (batch_size, num_masked_positions)
        labels :
            Shape (batch_size, seq_length)
        """
        if self._uniform_generator:
            # generate the corrupt tokens randomly with a mlm_scores vector whose value is all 0
            zero_logits = F.np.zeros(self.vocab_size)
            zero_logits = F.np.expand_dims(F.np.expand_dims(zero_logits, axis=0), axis=0)
            mlm_scores = F.np.expand_dims(F.np.zeros_like(masked_positions), axis=-1)
            mlm_scores = mlm_scores + zero_logits
        else:
            _, _, mlm_scores = self.generator(inputs, token_types, valid_length, masked_positions)

        corrupted_tokens, fake_data, labels = self.get_corrupted_tokens(
            F, inputs, unmasked_tokens, masked_positions, mlm_scores)
        # the discriminator take same input as the generator but the token_ids are
        # replaced with fake data
        _, _, rtd_scores = self.discriminator(fake_data, token_types, valid_length)
        return mlm_scores, rtd_scores, corrupted_tokens, labels

    def get_corrupted_tokens(self, F, inputs, unmasked_tokens, masked_positions, logits):
        """
        Sample from the generator to create corrupted input.
        Parameters
        ----------
        F
        inputs
            The masked input
            Shape (batch_size, seq_length)
        unmasked_tokens
            The original tokens that appear in the unmasked input sequence
            Shape (batch_size, num_masked_positions).
        masked_positions
            The masked position of the sequence
            Shape (batch_size, num_masked_positions).
        logits
            Shape (batch_size, num_masked_positions, vocab_size)

        Returns
        -------
        fake_data
            Shape (batch_size, seq_length)
        labels
            Shape (batch_size, seq_length)
        """

        if self._disallow_correct:
            disallow = F.npx.one_hot(masked_positions, depth=self.vocab_size, dtype=self._dtype)
            # TODO(zheyuye), Update when operation -= supported
            logits = logits - 1000.0 * disallow
        # gumbel_softmax() samples from the logits with a noise of Gumbel distribution
        prob = gumbel_softmax(
            F,
            logits,
            temperature=self._temperature,
            eps=1e-9,
            use_np_gumbel=False)
        corrupted_tokens = F.np.argmax(prob, axis=-1).astype(np.int32)

        # Following the Official electra to deal with duplicate positions as
        # https://github.com/google-research/electra/issues/41
        original_data, updates_mask = updated_vectors_by_position(F,
            inputs, unmasked_tokens, masked_positions)
        fake_data, _ = updated_vectors_by_position(F,
            inputs, corrupted_tokens, masked_positions)

        labels = updates_mask * F.np.not_equal(fake_data, original_data)
        return corrupted_tokens, fake_data, labels


def list_pretrained_electra():
    return sorted(list(PRETRAINED_URL.keys()))


def get_pretrained_electra(model_name: str = 'google_electra_small',
                           root: str = get_model_zoo_home_dir(),
                           load_backbone: bool = True,
                           load_disc: bool = False,
                           load_gen: bool = False) \
        -> Tuple[CN, HuggingFaceWordPieceTokenizer,
                 Optional[str],
                 Tuple[Optional[str], Optional[str]]]:
    """Get the pretrained Electra weights

    Parameters
    ----------
    model_name
        The name of the Electra model.
    root
        The downloading root
    load_backbone
        Whether to load the weights of the backbone network
    load_disc
        Whether to load the weights of the discriminator
    load_gen
        Whether to load the weights of the generator

    Returns
    -------
    cfg
        Network configuration
    tokenizer
        The HuggingFaceWordPieceTokenizer
    backbone_params_path
        Path to the parameter of the backbone network
    other_net_params_paths
        Path to the parameter of the discriminator and the generator.
        They will be returned inside a tuple.
    """
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}'.format(
        model_name, list_pretrained_electra())
    cfg_path = PRETRAINED_URL[model_name]['cfg']
    vocab_path = PRETRAINED_URL[model_name]['vocab']
    params_path = PRETRAINED_URL[model_name]['params']
    disc_params_path = PRETRAINED_URL[model_name]['disc_model']
    gen_params_path = PRETRAINED_URL[model_name]['gen_model']

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
    if load_disc:
        local_disc_params_path = download(url=get_repo_model_zoo_url() + disc_params_path,
                                          path=os.path.join(root, disc_params_path),
                                          sha1_hash=FILE_STATS[disc_params_path])
    else:
        local_disc_params_path = None

    if load_gen:
        local_gen_params_path = download(url=get_repo_model_zoo_url() + gen_params_path,
                                         path=os.path.join(root, gen_params_path),
                                         sha1_hash=FILE_STATS[gen_params_path])
    else:
        local_gen_params_path = None
    # TODO(sxjscience) Move do_lower to assets.
    tokenizer = HuggingFaceWordPieceTokenizer(
        vocab_file=local_paths['vocab'],
        unk_token='[UNK]',
        pad_token='[PAD]',
        cls_token='[CLS]',
        sep_token='[SEP]',
        mask_token='[MASK]',
        lowercase=True)
    cfg = ElectraModel.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_params_path, (local_disc_params_path, local_gen_params_path)


BACKBONE_REGISTRY.register('electra', [ElectraModel,
                                       get_pretrained_electra,
                                       list_pretrained_electra])
