""" Metrics for classification with soft (probabilistic) labels """

import logging

import mxnet as mx
import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-10  # clipping threshold to prevent NaN

# assumes predictions are already log-probabilities.
softloss = mx.gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False, from_logits=True)


def soft_log_loss(true_probs, predicted_probs):
    """ Both args must be 2D pandas/numpy arrays """
    true_probs = np.array(true_probs)
    predicted_probs = np.array(predicted_probs)
    if len(true_probs.shape) != 2 or len(predicted_probs.shape) != 2:
        raise ValueError("both truth and prediction must be 2D numpy arrays")
    if true_probs.shape != predicted_probs.shape:
        raise ValueError("truth and prediction must be 2D numpy arrays with the same shape")

    # true_probs = np.clip(true_probs, a_min=EPS, a_max=None)
    predicted_probs = np.clip(predicted_probs, a_min=EPS, a_max=None)  # clip 0s to avoid NaN
    true_probs = true_probs / true_probs.sum(axis=1, keepdims=1)  # renormalize
    predicted_probs = predicted_probs / predicted_probs.sum(axis=1, keepdims=1)
    losses = softloss(mx.nd.log(mx.nd.array(predicted_probs)), mx.nd.array(true_probs))
    return mx.nd.mean(losses).asscalar()
