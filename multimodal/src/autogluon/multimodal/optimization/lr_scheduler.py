"""
This file is largely borrowed from `transformers/optimization.py`.
This is to make lr schedulers pickle-able so that we can use the training strategy "ddp_spawn" in Pytorch Lightning.
"""
import functools
import math

from torch.optim.lr_scheduler import LambdaLR


def _cosine_decay_lr_lambda(current_step, num_warmup_steps, num_training_steps, num_cycles):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    lr_lambda = functools.partial(
        _cosine_decay_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _poly_decay_lr_lambda(
    current_step: int, num_warmup_steps: int, num_training_steps: int, lr_init: float, lr_end: float, power: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step > num_training_steps:
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init  # as LambdaLR multiplies by lr_init


def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.
    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    This function is borrowed from transformers/optimization.py. We make the function pickleble.

    Parameters
    ----------
    optimizer
        The optimizer for which to schedule the learning rate.
    num_warmup_steps
        The number of steps for the warmup phase.
    num_training_steps
        The total number of training steps.
    lr_end
        The end LR.
    power
        Power factor.
    last_epoch
        The index of the last epoch when resuming training.

    Returns
    -------
    lr_schedule
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init >= lr_end):
        raise ValueError(f"lr_end ({lr_end}) must not be larger than initial lr ({lr_init})")
    lr_lambda = functools.partial(
        _poly_decay_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_init=lr_init,
        lr_end=lr_end,
        power=power,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _linear_warmup_lr_lambda(current_step: int, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    lr_lambda = functools.partial(
        _linear_warmup_lr_lambda, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
