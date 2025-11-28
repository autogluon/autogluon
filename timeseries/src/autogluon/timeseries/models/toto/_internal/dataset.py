# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from functools import reduce
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from einops import repeat


def pad_array(
    values: torch.Tensor,
    patch_stride: int,
) -> torch.Tensor:
    """
    Makes sure that the series length is divisible by the patch_stride
    by adding left-padding.
    """
    if isinstance(values, np.ndarray):
        values = torch.from_numpy(values)
    series_len = values.shape[-1]
    # left-pad the time series to make sure we can divide it into patches.
    padded_length = int(np.ceil(series_len / patch_stride) * patch_stride)
    if values.ndim == 2:  # variates series_len
        padded_values = torch.zeros((values.shape[0], padded_length), dtype=values.dtype, device=values.device)
    elif values.ndim == 3:  # batch variates series_len
        padded_values = torch.zeros(
            (values.shape[0], values.shape[1], padded_length),
            dtype=values.dtype,
            device=values.device,
        )
    else:
        raise ValueError(f"Unsupported number of dimensions: {values.ndim}")
    padded_values[..., -series_len:] = values

    return padded_values


def pad_id_mask(
    id_mask: torch.Tensor,
    patch_stride: int,
) -> torch.Tensor:
    """
    Makes sure that the series length is divisible by the patch_stride
    by adding left-padding to the id mask. It does this by repeating
    the leftmost value of the id mask for each variate
    """
    series_len = id_mask.shape[-1]
    # left-pad the time series to make sure we can divide it into patches.
    padded_length = int(np.ceil(series_len / patch_stride) * patch_stride)
    padding_amount = padded_length - series_len
    left_edge: torch.Tensor = id_mask[..., 0]
    if id_mask.ndim == 2:  # variates series_len
        # repeat the left edge of the id mask for padding_amount
        padding = repeat(
            left_edge,
            "variates -> variates padding_amount",
            padding_amount=padding_amount,
        )
        id_mask = torch.cat([padding, id_mask], dim=1)
    elif id_mask.ndim == 3:  # batch variates series_len
        # repeat the left edge of the id mask for padding_amount
        padding = repeat(
            left_edge,
            "batch variates -> batch variates padding_amount",
            padding_amount=padding_amount,
        )
        id_mask = torch.cat([padding, id_mask], dim=2)
    else:
        raise ValueError(f"Unsupported number of dimensions: {id_mask.ndim}")

    return id_mask


class MaskedTimeseries(NamedTuple):
    series: torch.Tensor
    """
    The time series data, of shape (batch_size, num_variates, sequence_length). The first
    dimension is optional.
    """

    padding_mask: torch.Tensor
    """
    A mask that indicates which values are padding. If padding_mask[..., i] is True,
    then series[..., i] is _NOT_ padding; i.e., it's a valid value in the time series.
    Same shape as `series`.
    """

    id_mask: torch.Tensor
    """
    A mask that indicates the group ID of each variate. Any
    variates with the same ID are considered to be part of the same multivariate
    time series, and can attend to each other.

    Note: the sequence_length dimension can be 1 if the IDs should
    be broadcast across the time dimension.
    """

    timestamp_seconds: torch.Tensor
    """
    A POSIX timestamp in seconds for each time step in the series. Of same shape as 
    `series`.
    """

    time_interval_seconds: torch.Tensor
    """
    The time frequency of each variate in seconds. Of shape (batch_size, num_variates) with
    the first dimension optional.
    """

    def to(self, device: torch.device) -> "MaskedTimeseries":
        return MaskedTimeseries(
            series=self.series.to(device),
            padding_mask=self.padding_mask.to(device),
            id_mask=self.id_mask.to(device),
            timestamp_seconds=self.timestamp_seconds.to(device),
            time_interval_seconds=self.time_interval_seconds.to(device),
        )


def is_extreme_value(t: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(t):
        max_value = torch.finfo(t.dtype).max
    else:
        max_value = torch.iinfo(t.dtype).max

    return reduce(
        torch.logical_or,
        (
            torch.isinf(t),
            torch.isnan(t),
            t.abs() >= max_value / 2,
        ),
    )


def replace_extreme_values(t: torch.Tensor, replacement: float = 0.0) -> torch.Tensor:
    return torch.where(is_extreme_value(t), torch.tensor(replacement, dtype=t.dtype, device=t.device), t)


def freq_to_seconds(freq: str | pd.offsets.BaseOffset) -> float:
    # Modified from: https://github.com/DataDog/toto/blob/846d599f4b8d377db3088d5cd1a736d050cef5ac/toto/inference/gluonts_predictor.py#L58
    if isinstance(freq, str):
        freq = pd.tseries.frequencies.to_offset(freq)
    try:
        # Use nanos for fixed frequencies
        return freq.nanos / 1e9  # Convert nanoseconds to seconds
    except ValueError:
        # Handle non-fixed frequencies like Week
        if isinstance(freq, pd.offsets.BusinessDay):
            return freq.n * 24 * 60 * 60
        elif isinstance(freq, pd.offsets.Week):
            return freq.n * 7 * 24 * 60 * 60  # n weeks to seconds
        elif isinstance(freq, pd.offsets.MonthBegin) or isinstance(freq, pd.offsets.MonthEnd):
            return 30 * 24 * 60 * 60  # Approximate a month as 30 days
        elif isinstance(freq, pd.offsets.QuarterEnd) or isinstance(freq, pd.offsets.QuarterBegin):
            return 90 * 24 * 60 * 60  # Approximate a quarter as 90 days
        elif isinstance(freq, pd.offsets.YearEnd) or isinstance(freq, pd.offsets.YearBegin):
            return 365.25 * 24 * 60 * 60  # Approximate a year as 365.25 days
        else:
            raise ValueError(f"Cannot handle frequency of type {type(freq)}: {freq}")
