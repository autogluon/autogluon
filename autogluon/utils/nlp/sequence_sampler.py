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
"""Implements the beam search sampler."""
import numpy as np
import warnings
import mxnet as mx
import abc
from mxnet.gluon import HybridBlock
from typing import Callable, Optional
from .layers import get_activation


LARGE_POSITIVE_FLOAT = 1e18

LARGE_NEGATIVE_FLOAT = -LARGE_POSITIVE_FLOAT


class BaseStepDecoder(abc.ABC):
    """Base class of a step decoder

    You may inherit `BaseStepDecoder` and implement the required methods.

    """
    @property
    @abc.abstractmethod
    def state_batch_axis(self):
        """Batch axis of the state

        i --> axis of the batch dimension
        None --> no batch axis in the state
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def init_states(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, data, states):
        """The common signature of the sequence decoder

        Parameters
        ----------
        data
        states

        Returns
        -------
        out
        new_states
        """
        raise NotImplementedError


# TODO(sxjscience)
#  1. Add ParticleFilter Sampler
class BeamSearchScorer(HybridBlock):
    r"""Score function used in beam search.

    Implements the length-penalized score function first used in the GNMT paper::

        scores = (log_probs + scores) / length_penalty
        length_penalty = (\frac{K + length}{K + 1})^\alpha

    See Also

    "Google's Neural Machine Translation System: Bridging the Gap between Human and
    Machine Translation (https://arxiv.org/pdf/1609.08144.pdf)"

    Parameters
    ----------
    alpha
        If `alphas < 1.0`, it favors shorter sequences
        If `alpha >= 1.0`, it favors longer sequences
    K
        Parameter in the formula
    from_logits
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    temperature
        The temperature of the scoring function
    """
    def __init__(self, alpha: float = 1.0,
                 K: float = 5.0,
                 from_logits: bool = False,
                 temperature: float = 1.0,
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._alpha = float(alpha)
        self._K = K
        self._temperature = temperature
        self._from_logits = from_logits

    def __call__(self, outputs, scores, step):  # pylint: disable=arguments-differ
        """Compute new scores of each candidate

        Parameters
        ----------
        outputs : NDArray or Symbol
            If from_logits is True, outputs is the log probabilities of the candidates.
            Shape (d1, d2, ..., dn, V).
            Otherwise, outputs is the unnormalized outputs from predictor of the same shape,
            before softmax/log_softmax.
        scores : NDArray or Symbol
            The original scores of the beams. Shape (d1, d2, ..., dn)
        step : NDArray or Symbol
            Step to calculate the score function. It starts from 1. The shape is a scalar.

        Returns
        -------
        candidate_scores : NDArray or Symbol
            The scores of all the candidates. Shape (d1, d2, ..., dn, V), where V is the size
            of the vocabulary.
        """
        return super(BeamSearchScorer, self).__call__(outputs, scores, step)

    def hybrid_forward(self, F, outputs, scores, step):  # pylint: disable=arguments-differ
        if not self._from_logits:
            outputs = F.npx.log_softmax(outputs / self._temperature)

        if self._alpha != 0.0:
            step = step.astype(np.float32)
            prev_lp = (self._K + step - 1) ** self._alpha / ((self._K + 1) ** self._alpha)
            prev_lp = prev_lp * (step != 1).astype(np.float32) + (step == 1).astype(np.float32)
            lp = (self._K + step) ** self._alpha / ((self._K + 1) ** self._alpha)
            scores = scores * prev_lp
            candidate_scores = (outputs + F.np.expand_dims(scores, axis=-1)) / lp
        else:
            candidate_scores = outputs + F.np.expand_dims(scores, axis=-1)
        return candidate_scores

    def __repr__(self):
        s = '{name}(alpha={alpha}, K={K}, from_logits={from_logits})'
        return s.format(name=self.__class__.__name__,
                        alpha=self._alpha,
                        K=self._K,
                        from_logits=self._from_logits)


def _expand_to_beam_size(data, beam_size, batch_size, state_batch_axis=None):
    """Tile all the states to have batch_size * beam_size on the batch axis.

    Parameters
    ----------
    data : A single NDArray/Symbol or nested container with NDArrays/Symbol
        Each NDArray/Symbol should have shape (N, ...) when state_info is None,
        or same as the layout in state_info when it's not None.
    beam_size : int
        Beam size
    batch_size : int
        Batch size
    state_batch_axis : Nested structure of dictionary, default None.
        Descriptors for states, usually from decoder's ``state_batch_axis()``.
        When None, this method assumes that the batch axis is the first dimension.
    Returns
    -------
    new_states : Object that contains NDArrays/Symbols
        Each NDArray/Symbol should have shape batch_size * beam_size on the batch axis.
    """
    if isinstance(data, (list, tuple)):
        if state_batch_axis is not None:
            # TODO(sxjscience) Better Exception Handling
            return [_expand_to_beam_size(d, beam_size, batch_size, batch_axis)
                    for d, batch_axis in zip(data, state_batch_axis)]
        else:
            return [_expand_to_beam_size(d, beam_size, batch_size, None) for d in data]
    elif isinstance(data, dict):
        if state_batch_axis is not None:
            return {k: _expand_to_beam_size(v, beam_size, batch_size, state_batch_axis[k])
                    for k, v in data.items()}
        else:
            return {k: _expand_to_beam_size(v, beam_size, batch_size, None)
                    for k, v in data.items()}
    elif isinstance(data, mx.np.ndarray):
        if state_batch_axis is None:
            batch_axis = 0
        else:
            batch_axis = state_batch_axis
        if data.shape[batch_axis] != batch_size:
            raise ValueError('The batch size of all the inner elements in states must be '
                             '{}, Found shape={}, inferred batch axis={}'.format(batch_size, data.shape, batch_axis))
        new_shape = list(data.shape)
        new_shape[batch_axis] = batch_size * beam_size
        new_shape = tuple(new_shape)
        bcast_new_shape = new_shape[:batch_axis] + (batch_size, beam_size) + new_shape[(batch_axis + 1):]
        return mx.np.expand_dims(data, batch_axis + 1).broadcast_to(bcast_new_shape).reshape(new_shape)
    elif isinstance(data, mx.sym.Symbol):
        raise NotImplementedError
    elif data is None:
        return None
    else:
        raise NotImplementedError


def _choose_states(F, states, indices, state_batch_axis=None):
    """

    Parameters
    ----------
    F : ndarray or symbol
    states : Object contains NDArrays/Symbols
    indices : NDArray or Symbol
        Indices of the states to take. Shape (N,).
    state_batch_axis
        Descriptors for states, it is generated from decoder's ``state_batch_axis``.
        When None, this method assumes that the batch axis is the first dimension.

    Returns
    -------
    new_states : Object contains NDArrays/Symbols
        Each NDArray/Symbol should have shape (..., N, ...).
    """
    if isinstance(states, (list, tuple)):
        if state_batch_axis is not None:
            return [_choose_states(F, d, indices, b_axis)
                    for d, b_axis in zip(states, state_batch_axis)]
        else:
            return [_choose_states(F, d, indices, None) for d in states]
    elif isinstance(states, dict):
        if state_batch_axis is not None:
            return {k: _choose_states(F, v, indices, state_batch_axis[k]) for k, v in states.items()}
        else:
            return {k: _choose_states(F, v, indices, None) for k, v in states.items()}
    elif isinstance(states, (mx.np.ndarray, mx.sym.numpy._Symbol)):
        if state_batch_axis is None:
            batch_axis = 0
        else:
            batch_axis = state_batch_axis
        states = F.np.take(states, indices, axis=batch_axis)
        return states
    else:
        raise TypeError('The type of the states is not supported, type(states) = {}'.format(type(states)))


class _BeamSearchStepUpdate(HybridBlock):
    def __init__(self, beam_size, vocab_size, eos_id, scorer, state_batch_axis,
                 stochastic=False, prefix=None, params=None):
        """

        Parameters
        ----------
        beam_size : int
        vocab_size : int
        eos_id : int
        scorer : BeamSearchScorer
        state_batch_axis :
        stochastic: bool
        prefix : None
        params : None
        """
        super(_BeamSearchStepUpdate, self).__init__(prefix=prefix, params=params)
        self._beam_size = beam_size
        self._vocab_size = vocab_size
        self._eos_id = eos_id
        self._scorer = scorer
        self._state_batch_axis = state_batch_axis
        self.stochastic = stochastic
        self.activation = get_activation('relu')
        assert eos_id >= 0, 'eos_id cannot be negative! Received eos_id={}'.format(eos_id)

    def gumbel_with_maximum(self, F, phi, T, dim=-1):
        """
        Parameters
        ----------
        F
        phi : mx.np.ndarray or Symbol
            Shape (batch_size, beam_size, L).
        T : mx.np.ndarray or Symbol
            The previous scores. Shape (batch_size, beam_size)
        """
        g_phi = phi + F.np.random.gumbel(F.np.zeros_like(phi))
        Z = g_phi.max(dim)
        g = self.shift_gumbel_maximum(F, g_phi, T, dim, Z=Z)
        return g
    
    def shift_gumbel_maximum(self, F, g_phi, T, dim=-1, Z=None):
        """
        Parameters
        ----------
        F
        g_phi : mx.np.ndarray or Symbol
            Shape (batch_size, beam_size, L).
        T : mx.np.ndarray or Symbol
            The previous scores. Shape (batch_size, beam_size)
        """
        if Z is None:
            Z = g_phi.max(dim)
        T_ = F.npx.reshape(T, (-4, 1))
        Z_ = F.npx.reshape(Z, (-4, 1))
        u = T_ - g_phi + F.np.log1p(-F.np.exp(g_phi - Z_)+1e-5)
        return T_ - self.activation(u) - F.np.log1p(F.np.exp(-F.np.abs(u)))

    def hybrid_forward(self, F, samples, valid_length, outputs, scores, step, beam_alive_mask,   # pylint: disable=arguments-differ
                       states, batch_shift):
        """

        Parameters
        ----------
        F
        samples : mx.np.ndarray or Symbol
            The current samples generated by beam search.
            Shape (batch_size, beam_size, L).
        valid_length : NDArray or Symbol
            The current valid lengths of the samples
        outputs : NDArray or Symbol
            Outputs from predictor. If from_logits was set to True in scorer, then it's the
            log probability of the current step. Else, it's the unnormalized outputs before
            softmax or log_softmax.
            Shape (batch_size * beam_size, V).
        scores : NDArray or Symbol
            The previous scores. Shape (batch_size, beam_size)
        step : NDArray or Symbol
            The current step for doing beam search. Begins from 1. Shape ()
        beam_alive_mask : NDArray or Symbol
            Shape (batch_size, beam_size)
        states : nested structure of NDArrays/Symbols
            Each NDArray/Symbol should have shape (N, ...) when state_info is None,
            or same as the layout in state_info when it's not None.
        batch_shift : NDArray or Symbol
            Contains [0, beam_size, 2 * beam_size, ..., (batch_size - 1) * beam_size].
            Shape (batch_size,)

        Returns
        -------
        new_samples : NDArray or Symbol or an empty list
            The updated samples.
            When single_step is False, shape (batch_size, beam_size, L + 1)
        new_valid_length : NDArray or Symbol
            Valid lengths of the samples. Shape (batch_size, beam_size)
        new_scores : NDArray or Symbol
            Shape (batch_size, beam_size)
        chosen_word_ids : NDArray or Symbol
            The chosen word ids of the step. Shape (batch_size, beam_size). If it's negative,
            no word will be appended to the beam.
        beam_alive_mask : NDArray or Symbol
            Shape (batch_size, beam_size)
        new_states : nested structure of NDArrays/Symbols
            Inner NDArrays have shape (batch_size * beam_size, ...)
        """
        beam_size = self._beam_size
        vocab_size = self._vocab_size
        beam_alive_mask_bcast = F.np.expand_dims(beam_alive_mask, axis=2)
        candidate_scores = self._scorer(F.npx.reshape(outputs, (-6, -1, beam_size, -2)),
                                        scores, step)
        if self.stochastic:
            if step == 1:
                candidate_scores_gumbel\
                    = candidate_scores[:1]\
                      + F.np.random.gumbel(F.np.zeros_like(candidate_scores[:1]))
                candidate_scores_residual = candidate_scores[1:]
                candidate_scores = F.np.concatenate((candidate_scores_gumbel,
                                                     candidate_scores_residual), axis=0)
            else:
                candidate_scores = self.gumbel_with_maximum(F, candidate_scores, scores, -1)
        # Concat the candidate scores and the scores of the finished beams
        # The resulting candidate score will have shape (batch_size, beam_size * |V| + beam_size)
        candidate_scores = F.np.where(beam_alive_mask_bcast,
                                      candidate_scores,
                                      F.np.full_like(candidate_scores,
                                                     LARGE_NEGATIVE_FLOAT))
        finished_scores = F.np.where(beam_alive_mask,
                                     F.np.full_like(scores,
                                                    LARGE_NEGATIVE_FLOAT),
                                     scores)
        candidate_scores = F.np.concatenate([F.npx.reshape(candidate_scores, (-2, -1)),
                                             finished_scores],
                                            axis=1)
        # Get the top K scores
        # new_scores and indices will have shape (batch_size, beam_size)
        new_scores, indices = F.npx.topk(candidate_scores, axis=1, k=beam_size, ret_typ='both')
        indices = indices.astype(np.int32)
        use_prev = (indices >= (beam_size * vocab_size)).astype(np.int32)
        chosen_word_ids = F.np.mod(indices, vocab_size)
        beam_ids = F.np.where(use_prev, indices - beam_size * vocab_size,
                              F.np.floor(indices / vocab_size).astype(np.int32))
        batch_beam_indices = beam_ids + F.np.expand_dims(batch_shift, axis=1)
        chosen_word_ids = F.np.where(use_prev, - F.np.ones_like(indices), chosen_word_ids)
        # Update the samples and vaild_length
        # TODO(sxjscience) The current implementation is quite tricky
        #  We should wait for hybridizable advanced indexing to avoid this
        selected_samples = F.np.take(F.npx.reshape(samples, (-5, -2)),
                                     batch_beam_indices.reshape((-1,)), axis=0)
        new_samples = F.npx.reshape(F.np.concatenate([selected_samples,
                                                      chosen_word_ids.reshape((-1, 1))],
                                                     axis=1),
                                    (-6, -1, beam_size, -2))
        new_valid_length = F.np.take(valid_length.reshape((-1,)),
                                     batch_beam_indices.reshape((-1,)),
                                     axis=0).reshape((-1, beam_size)) + 1 - use_prev
        # Update the states
        new_states = _choose_states(F, states, batch_beam_indices.reshape((-1,)),
                                    self._state_batch_axis)
        # Update the alive mask.
        beam_alive_mask = F.np.take(beam_alive_mask.reshape((-1,)),
                                    batch_beam_indices.reshape((-1,)), axis=0)\
                              .reshape((-1, beam_size))\
                          * (chosen_word_ids != self._eos_id).astype(np.float32)
        return new_samples, new_valid_length, new_scores, chosen_word_ids,\
               beam_alive_mask, new_states


class BeamSearchSampler:
    r"""Draw samples from the decoder by beam search.

    Parameters
    ----------
    beam_size : int
        The beam size.
    decoder
        Function of the one-step-ahead decoder, should inherit SequenceDecoder and
        have the form::

            outputs, new_states = decoder(step_input, states)

        The outputs, input should follow these rules:

        - step_input has shape (batch_size,),
        - outputs has shape (batch_size, V),
        - states and new_states have the same structure.
    eos_id
        Id of the EOS token. No other elements will be appended to the sample if it reaches eos_id,
        if eos_id == None, beam will not die
    scorer : BeamSearchScorer, default BeamSearchScorer(alpha=1.0, K=5)
        The score function used in beam search.
    max_length_a
        TODO(sxjscience) We can potentially make it more general.
        The `a` value in the formula `a * x + b`. Generate sequences of maximum length `a * x + b`,
        where `x` is the maximum source length.
    max_length_b
        The b value in the formula `a * x + b`. Generate sequences of maximum length `a * x + b`,
        where `x` is the maximum source length.
    min_length
        The minimum length of the generated sequences.
    temperature
    stochastic
        Whether to use stochastic sampler,
        see [ICML2019] "Stochastic Beams and Where to Find Them" for detail.
        https://arxiv.org/abs/1903.06059
    sampling
        Whether to use multinomial sampler.
    sampling_topp
        Multinomial sampling with topp,
        see [ICLR2020] "The Curious Case of Neural Text Degeneration"
        https://arxiv.org/abs/1904.09751
    sampling_topk
        Multinomial sampling with topk,
        see [ACL2018] "Hierarchical Neural Story Generation"
        https://www.aclweb.org/anthology/P18-1082.pdf
    """
    def __init__(self, beam_size: int,
                 decoder: BaseStepDecoder,
                 eos_id: int,
                 vocab_size: int,
                 scorer: Optional[Callable] = None,
                 max_length_a: int = 0,
                 max_length_b: int = 200,
                 min_length: int = 1,
                 temperature: float = 1.0,
                 stochastic: bool = False,
                 sampling: bool = False,
                 sampling_topp: float = -1.0,
                 sampling_topk: int = -1):
        self._beam_size = beam_size
        self._vocab_size = vocab_size
        assert beam_size > 0,\
            'beam_size must be larger than 0. Received beam_size={}'.format(beam_size)
        self._decoder = decoder
        self._eos_id = eos_id
        assert eos_id >= 0, 'eos_id cannot be negative! Received eos_id={}'.format(eos_id)
        self._max_length_a = max_length_a
        self._max_length_b = max_length_b
        self._min_length = min_length
        if scorer is None:
            if stochastic:
                scorer = BeamSearchScorer(alpha=0.0, temperature=temperature)
            else:
                scorer = BeamSearchScorer(alpha=1.0, K=5, temperature=temperature)

        self._scorer = scorer
        self._state_batch_axis = decoder.state_batch_axis
        self._sampling = sampling
        self._sampling_topp = sampling_topp
        self._sampling_topk = sampling_topk
        if sampling:
            self._updater = _MultinomialStepUpdate(
                beam_size=beam_size,
                vocab_size=vocab_size,
                eos_id=eos_id,
                state_batch_axis=decoder.state_batch_axis,
                sampling_topp=sampling_topp,
                sampling_topk=sampling_topk,
                temperature=temperature
            )
        else:
            self._updater = _BeamSearchStepUpdate(
                beam_size=beam_size,
                vocab_size=vocab_size,
                eos_id=eos_id,
                scorer=scorer,
                state_batch_axis=decoder.state_batch_axis,
                stochastic=stochastic
            )

        if not stochastic:
            self._updater.hybridize()
        else:
            if isinstance(scorer, BeamSearchScorer):
                if scorer._alpha != 0.0:
                    warnings.warn(
                        'To use stochastic beam search, we need to set the alpha as 0.0')

    def __call__(self, inputs, states, src_seq_lengths=None, early_return=True):
        """Sample by beam search.

        Parameters
        ----------
        inputs : NDArray
            The initial input of the decoder. Shape is (batch_size,).
        states : Object that contains NDArrays
            The initial states of the decoder.
        src_seq_lengths : NDArray
            The source sequence lengths. Shape is (batch_size,).
        early_return : bool
            Whether to return when all beams are dead.
            Without early_return, the sequences will be generated until the
            maximum length is reached.

        Returns
        -------
        samples : NDArray
            Samples draw by beam search. Shape (batch_size, beam_size, length).
            DType is int32.
        scores : NDArray
            Scores of the samples. Shape (batch_size, beam_size).
             We make sure that scores[i, :] are in descending order.
        valid_length : NDArray
            The valid length of the samples. Shape (batch_size, beam_size).
            DType is int32.
        """
        ctx = inputs.ctx
        batch_size = inputs.shape[0]
        beam_size = self._beam_size
        if src_seq_lengths is not None:
            max_src_sequence_length = int(src_seq_lengths.asnumpy().max())
            max_length = max(self._min_length, max_src_sequence_length * self._max_length_a
                             + self._max_length_b)
        else:
            if self._max_length_a != 0:
                raise ValueError('If src_seq_lengths is not given, max_length_a must be 0!'
                                 ' Received {}'
                                 .format(self._max_length_a))
            max_length = max(self._min_length, self._max_length_b)
        # Tile the states and inputs to have shape (batch_size * beam_size, ...)
        states = _expand_to_beam_size(states, beam_size=beam_size, batch_size=batch_size,
                                      state_batch_axis=self._state_batch_axis)
        step_input = _expand_to_beam_size(inputs, beam_size=beam_size,
                                          batch_size=batch_size).astype(np.int32)
        # All beams are initialized to alive
        # Generated samples are initialized to be the inputs
        # Except the first beam where the scores are set to be zero, all beams have -inf scores.
        # Valid length is initialized to be 1
        beam_alive_mask = mx.np.ones(shape=(batch_size, beam_size), ctx=ctx, dtype=np.float32)
        valid_length = mx.np.ones(shape=(batch_size, beam_size), ctx=ctx, dtype=np.int32)
        scores = mx.np.zeros(shape=(batch_size, beam_size), ctx=ctx)
        if beam_size > 1:
            scores[:, 1:beam_size] = LARGE_NEGATIVE_FLOAT
        samples = step_input.reshape((batch_size, beam_size, 1))
        batch_shift = mx.np.arange(0, batch_size * beam_size, beam_size, ctx=ctx, dtype=np.int32)
        step = mx.np.array(0, ctx=ctx, dtype=np.float32)
        for i in range(max_length):
            log_probs, new_states = self._decoder(step_input, states)
            assert log_probs.shape[1] == self._vocab_size
            step = step + 1
            samples, valid_length, scores, chosen_word_ids, beam_alive_mask, states = \
                self._updater(samples, valid_length, log_probs, scores, step, beam_alive_mask,
                              new_states, batch_shift)
            step_input = mx.npx.relu(chosen_word_ids).reshape((-1,))
            if early_return:
                if mx.np.sum(beam_alive_mask).asnumpy() == 0:
                    return samples, scores, valid_length
        beam_alive_mask = beam_alive_mask.astype(np.int32)
        final_word = mx.np.where(beam_alive_mask,
                                 mx.np.full((batch_size, beam_size), self._eos_id,
                                            ctx=ctx, dtype=np.int32),
                                 mx.np.full((batch_size, beam_size), -1, ctx=ctx, dtype=np.int32))
        samples = mx.np.concatenate([samples,
                                     final_word.reshape((final_word.shape[0],
                                                         final_word.shape[1], 1))],
                                    axis=2)
        valid_length += beam_alive_mask
        return samples, scores, valid_length

    def __repr__(self):
        ret = '{name}:(\n' \
              '  beam_size={beam_size}\n' \
              '  eos_id={eos_id}\n' \
              '  vocab_size={vocab_size}\n' \
              '  max_length_a={max_length_a}\n' \
              '  max_length_b={max_length_b}\n' \
              '  scorer={scorer}\n' \
              '  sampling={sampling}\n' \
              '  sampling_topp={sampling_topp}\n' \
              '  sampling_topk={sampling_topk}\n' \
              ')' \
            .format(name=self.__class__.__name__,
                    beam_size=self._beam_size,
                    eos_id=self._eos_id,
                    vocab_size=self._vocab_size,
                    max_length_a=self._max_length_a,
                    max_length_b=self._max_length_b,
                    scorer=self._scorer,
                    sampling=self._sampling,
                    sampling_topp=self._sampling_topp,
                    sampling_topk=self._sampling_topk)
        return ret

class _MultinomialStepUpdate(HybridBlock):
    def __init__(self, beam_size, vocab_size, eos_id, state_batch_axis,
                 sampling_topp=-1.0, sampling_topk=-1, temperature=1.0,
                 prefix=None, params=None):
        super(_MultinomialStepUpdate, self).__init__(prefix=prefix, params=params)
        self._beam_size = beam_size
        self._vocab_size = vocab_size
        self._eos_id = eos_id
        self._state_batch_axis = state_batch_axis
        self._sampling_topp = sampling_topp
        self._sampling_topk = sampling_topk
        self._temperature = temperature
        self.activation = get_activation('relu')
        assert eos_id >= 0, 'eos_id cannot be negative! Received eos_id={}'.format(eos_id)
        assert sampling_topp <= 0 or sampling_topk <= 0, 'sampling_topp conflicts with sampling_topk'

    def hybrid_forward(self, F, samples, valid_length, outputs, scores, step, beam_alive_mask,
                       states, batch_shift):
        """

        Parameters
        ----------
        F
        samples : mx.np.ndarray or Symbol
            The current samples generated by beam search.
            Shape (batch_size, beam_size, L).
        valid_length : NDArray or Symbol
            The current valid lengths of the samples
        outputs : NDArray or Symbol
            Outputs from predictor. If from_logits was set to True in scorer, then it's the
            log probability of the current step. Else, it's the unnormalized outputs before
            softmax or log_softmax.
            Shape (batch_size * beam_size, V).
        scores : NDArray or Symbol
            The previous scores. Shape (batch_size, beam_size)
        step : NDArray or Symbol
            The current step for doing beam search. Begins from 1. Shape ()
        beam_alive_mask : NDArray or Symbol
            Shape (batch_size, beam_size)
        states : nested structure of NDArrays/Symbols
            Each NDArray/Symbol should have shape (N, ...) when state_info is None,
            or same as the layout in state_info when it's not None.
        batch_shift : NDArray or Symbol
            Contains [0, beam_size, 2 * beam_size, ..., (batch_size - 1) * beam_size].
            Shape (batch_size,)

        Returns
        -------
        new_samples : NDArray or Symbol or an empty list
            The updated samples.
            When single_step is False, shape (batch_size, beam_size, L + 1)
        new_valid_length : NDArray or Symbol
            Valid lengths of the samples. Shape (batch_size, beam_size)
        new_scores : NDArray or Symbol
            Shape (batch_size, beam_size)
        chosen_word_ids : NDArray or Symbol
            The chosen word ids of the step. Shape (batch_size, beam_size). If it's negative,
            no word will be appended to the beam.
        beam_alive_mask : NDArray or Symbol
            Shape (batch_size, beam_size)
        new_states : nested structure of NDArrays/Symbols
            Inner NDArrays have shape (batch_size * beam_size, ...)
        """
        # bsz * beam_size * vocab_size
        outputs = outputs.reshape((-1, self._beam_size, self._vocab_size))
        probs = F.npx.softmax(outputs / self._temperature)
        
        if self._sampling_topp > 0:
            probs = F.np.where(
                probs > self._sampling_topp,
                probs,
                F.np.zeros_like(probs)
            )
        elif self._sampling_topk > 0:
            topk_probs = F.npx.topk(probs, axis=2, k=self._sampling_topk, ret_typ='value')
            # choose the k max prob
            k_prob = topk_probs[:,:,-1]
            k_prob = F.np.expand_dims(k_prob, axis=-1)
            probs = F.np.where(
                probs >= k_prob,
                probs,
                F.np.zeros_like(probs)
            )
            
        # renormalize
        probs_sum = F.np.sum(probs, axis=2, keepdims=True)
        probs = probs / probs_sum
        
        # bsz * beam_size
        chosen_word_ids, chosen_word_log_probs = \
            F.npx.random.categorical(probs, get_prob=True)
        
        new_scores = scores + F.np.where(
            beam_alive_mask,
            chosen_word_log_probs,
            F.np.zeros_like(chosen_word_log_probs)
        )
        
        # mask dead words
        chosen_word_ids = F.np.where(
            beam_alive_mask,
            chosen_word_ids,
            F.np.full_like(beam_alive_mask, -1, dtype=np.int32)
        )

        new_valid_length = valid_length + beam_alive_mask.astype(np.int32)
        new_samples = F.np.concatenate(
            [samples, F.np.expand_dims(chosen_word_ids, axis=2)],
            axis=2
        )
        new_states = states
        beam_alive_mask = beam_alive_mask * (chosen_word_ids != self._eos_id).astype(np.int32)
        
        return new_samples, new_valid_length, new_scores, chosen_word_ids,\
               beam_alive_mask, new_states
