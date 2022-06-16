import numpy as np
import torch
from timm.data.mixup import Mixup, mixup_target, cutmix_bbox_and_lam


class MixupModule(Mixup):
    """
    Mixup class from timm.
    https://github.com/rwightman/pytorch-image-models/blob/d30685c283137b4b91ea43c4e595c964cd2cb6f0/timm/data/mixup.py
    The parameters here are correspond to the mixup config in data.
    The mixup in timm only produce image mixup and cutmix with one-hot class target.
    This module helps to take the lambda from the Mixup.
    Lambda is added to the function to produce the mixup with specific lambda.
    """

    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        correct_lam=True,
        label_smoothing=0.1,
        num_classes=1000,
    ):
        """
        Parameters
        ----------
        mixup_alpha
            The mixup alpha value, it is active if > 0.
        cutmix_alpha
            The cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax
            cutmix min/max image ratio. The para should be a list/tuple of float with size 2.
        prob
            The probability of conducting mixup/cutmix if enable.
        switch_prob
            The probability of switching mixup to cutmix if both enable.
        mode
            Perform mixup/cutmix on "batch" or "pair" or "elem".
        correct_lam
            Apply lambda correction when cutmix bbox clipped by image borders.
        label_smoothing
            Apply label smoothing to the mixed target.
        num_classes
            Number of classes for target.
        """
        super().__init__(
            mixup_alpha,
            cutmix_alpha,
            cutmix_minmax,
            prob,
            switch_prob,
            mode,
            correct_lam,
            label_smoothing,
            num_classes,
        )
        self.lam = None
        self.target_a = None
        self.target_b = None

    def _mix_elem(self, x, lam_batch):
        batch_size = len(x)
        if lam_batch is None:
            lam_batch, use_cutmix = self._params_per_elem(batch_size)
        else:
            _, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone()
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.0:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam
                    )
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_pair(self, x, lam_batch):
        batch_size = len(x)
        if lam_batch is None:
            lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        else:
            _, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.0:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam
                    )
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_batch(self, x, lam):
        if lam is None:
            lam, use_cutmix = self._params_per_batch()
        else:
            _, use_cutmix = self._params_per_batch()
        if lam == 1.0:
            return 1.0
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam
            )
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1.0 - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target, lam=None):
        if self.mode == "elem":
            lam = self._mix_elem(x, lam)
        elif self.mode == "pair":
            lam = self._mix_pair(x, lam)
        else:
            lam = self._mix_batch(x, lam)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device)
        return x, target, lam


def mixup_others(x, lam):
    """
    Mixup special types of data, espcially for tuple.
    It is the simplest way of mixup for non image data.
    If lam >=0.5: choose the origin, else: choose the other one.

    Parameters
    -------
    x
        The target need to be mixed-up.
    lam
        The mixup lambda.

    Returns
    -------
    The mixed-up batch data with specific model.
    """
    if lam is None:
        lam = 1
    else:
        lam = round(lam)
    if isinstance(x, tuple):
        target = (pertarget * lam + pertarget.flip(0) * (1.0 - lam) for pertarget in x)
    else:
        target = x * lam + x.flip(0) * (1.0 - lam)
    return target


def multimodel_mixup(batch, model, mixup_fn):
    """
    Mixup for different models.
    For image data, use the mixup_fn from timm.
    For other types of data, the simplest way as choosing will be used.

    Parameters
    -------
    batch
        The origin data need to be mixed-up.
    model
        The model used on the task.It is used to get the useful column in batch.
    mixup_fn
        The mixup_fn from timm. It can mixup image and produce target label with lambda.

    Returns
    -------
    batch
        The mixed-up batch.
    mixup_label
        The mixed-up labels.
    """
    mixup_label = batch[model.label_key]
    if hasattr(model, "image_key"):
        batch[model.image_key], mixup_label = mixup_fn(batch[model.image_key], batch[model.label_key])
    else:
        lam = None
        for permodel in model.model:
            if hasattr(permodel, "image_key"):
                if lam is None:
                    batch[permodel.image_key], mixup_label, lam = mixup_fn(
                        batch[permodel.image_key], batch[permodel.label_key]
                    )
                else:
                    batch[permodel.image_key], _, _ = mixup_fn(
                        batch[permodel.image_key], batch[permodel.label_key], lam
                    )
        for permodel in model.model:
            if hasattr(permodel, "categorical_key"):
                mixup_others(batch[permodel.categorical_key], lam)
            if hasattr(permodel, "numerical_key"):
                mixup_others(batch[permodel.numerical_key], lam)
            if hasattr(permodel, "text_token_ids_key"):
                mixup_others(batch[permodel.text_token_ids_key], lam)

    return batch, mixup_label
