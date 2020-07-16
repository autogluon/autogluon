from mxnet.gluon import HybridBlock


class LabelSmoothCrossEntropyLoss(HybridBlock):
    r"""Computes the softmax cross entropy loss with label-smoothing

    .. math::

        \DeclareMathOperator{softmax}{softmax}

        lp = \log \softmax({pred})

        y = (1 - \alpha) one_hot({label}) + \frac{\alpha}{N}
        L = - \sum_{i=1}^N y_i lp_i

    To reduce complexity, we can implement it as

    .. math::

        L_i = - [(1 - \alpha) lp_{i, {label}_i}) + \alpha \frac{1}{N} \sum_{j=1}^N (lp_{i, j})]

    Parameters
    ----------
    num_labels
        The number of possible labels. For example, in NLP, it can be the size of the vocabulary.
    alpha
        The uncertainty that will be injected to the labels. All the negative labels will be
        treated with probability equals to \frac{\alpha} / {N}
    from_logits
        Whether input is a log probability (usually from log_softmax) instead of unnormalized numbers.
    """
    def __init__(self, num_labels: int, alpha: float = 0.1, from_logits: bool = False, **kwargs):
        super(LabelSmoothCrossEntropyLoss, self).__init__(**kwargs)
        self._num_labels = num_labels
        self._alpha = alpha
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label):
        """

        Parameters
        ----------
        F
        pred :
            The predictions of the network. Shape (..., V)
        label :
            The labels. Shape (..., )

        Returns
        -------
        loss :
            Shape (..., )
        """
        if not self._from_logits:
            pred = F.npx.log_softmax(pred, axis=-1)
        log_likelihood = F.npx.pick(pred, label, axis=-1)
        all_scores = pred.sum(axis=-1)
        loss = - (1 - self._alpha) * log_likelihood\
               - self._alpha / float(self._num_labels) * all_scores
        return loss
