from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple, Union

Point = Tuple[int, int]
PointsSpec = Union[Iterable[Point], List[Union[Point, None]]]


def _interp_loglog(n: float, n0: float, y0: float, n1: float, y1: float) -> float:
    """
    Power-law interpolation:
        log(y) linear in log(n)
    """
    ln = math.log10(n)
    l0 = math.log10(n0)
    l1 = math.log10(n1)

    ly0 = math.log10(y0)
    ly1 = math.log10(y1)

    t = (ln - l0) / (l1 - l0)
    return 10 ** (ly0 + t * (ly1 - ly0))


def _parse_points_spec(points: PointsSpec) -> tuple[list[Point], bool]:
    """
    Returns (anchors, tail_none).
    tail_none=True means "above last anchor -> None".
    """
    seq = list(points)

    tail_none = bool(seq) and (seq[-1] is None)
    if tail_none:
        seq = seq[:-1]

    if not seq:
        raise ValueError("points must contain at least one (num_samples_val, max_models) anchor.")

    anchors: list[Point] = []
    for p in seq:
        if p is None:
            raise ValueError("Only a trailing None is allowed (e.g., [..., None]).")
        n, m = p
        n = int(n)
        if n <= 0:
            raise ValueError(f"Anchor num_samples_val must be > 0, got {n}.")
        if m is not None and int(m) < 0:
            raise ValueError(f"Anchor max_models must be >= 0, got {m}.")
        anchors.append((n, int(m)))

    # Validate increasing n
    for (n0, _), (n1, _) in zip(anchors, anchors[1:]):
        if n1 <= n0:
            raise ValueError("Anchor num_samples_val must be strictly increasing.")

    return anchors, tail_none


def max_models_from_num_samples_val(
    num_samples_val: int,
    points: PointsSpec = ((100, 3), (10_000, 25), (100_000, 100), None),
    *,
    rounding: str = "floor",  # "round" | "floor" | "ceil"
) -> Optional[int]:
    """
    Smooth monotone map from validation size -> max_models using:
      - log10(num_samples_val) for x-axis
      - smoothstep easing within each segment
      - piecewise interpolation between successive anchors

    If points ends with None, returns None for num_samples_val > last_anchor_n.
    """
    anchors, tail_none = _parse_points_spec(points)

    n = int(num_samples_val)
    if n <= 0:
        # conservative default: first anchor's value
        return anchors[0][1]

    # Tail behavior
    last_n, last_m = anchors[-1]
    if tail_none and n > last_n:
        return None

    # Clamp below first anchor
    first_n, first_m = anchors[0]
    if n <= first_n:
        return first_m

    # Find segment [i, i+1] such that anchors[i].n < n <= anchors[i+1].n
    for (n0, y0), (n1, y1) in zip(anchors, anchors[1:]):
        if n <= n1:
            y = _interp_loglog(n, n0, y0, n1, y1)

            if rounding == "floor":
                out = int(math.floor(y))
            elif rounding == "ceil":
                out = int(math.ceil(y))
            else:
                out = int(round(y))
            lo, hi = sorted((y0, y1))
            return max(lo, min(hi, out))

    # If we got here: n > last_n and tail_none is False => clamp to last
    return last_m
