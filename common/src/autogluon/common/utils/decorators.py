import functools
import logging
from typing import Dict
from urllib.parse import urlparse

from .presets_io import load_preset_dict_from_location

logger = logging.getLogger(__name__)


def unpack(g, *other_args):
    """
    Used to pass *args and **kwargs to a function g prior to entering function f.

    Examples
    --------

    >>> def g(a=0, **kwargs):
    >>>     kwargs['b'] = a + 1
    >>>     return kwargs
    >>>
    >>> @unpack(g)
    >>> def f(**kwargs):
    >>>     print(kwargs)
    >>>
    >>> f(a=2)  # kwargs is now the output of g(a=2), which is {'b': 3}
    >>> f(c=4)  # kwargs is now the output of g(c=4), which is {'b': 1, 'c': 4}
    """

    def _unpack_inner(f):
        @functools.wraps(f)
        def _call(*args, **kwargs):
            gargs, gkwargs = g(*other_args, *args, **kwargs)
            return f(*gargs, **gkwargs)

        return _call

    return _unpack_inner


def _looks_like_preset_location(s: str) -> bool:
    """
    Return True only if `s` clearly refers to a file or URL.
    Conservative by design to preserve old error behavior.
    """
    # Explicit schemes
    parsed = urlparse(s)
    if parsed.scheme in {"s3", "http", "https", "file"}:
        return True

    # Local file path heuristics
    if s.endswith((".yaml", ".yml")):
        return True

    # Relative/absolute paths
    if s.startswith(("./", "../", "/")):
        return True

    return False


def _resolve_preset_str(
    preset_og: str,
    preset_dict: Dict[str, dict],
    presets_alias: Dict[str, str] | None,
) -> dict:
    # 1) Built-in preset
    preset = preset_dict.get(preset_og)
    if preset is not None:
        return preset

    # 2) Alias
    if presets_alias is not None:
        mapped = presets_alias.get(preset_og)
        if mapped is not None:
            logger.log(20, f"Preset alias specified: '{preset_og}' maps to '{mapped}'.")
            preset = preset_dict.get(mapped)
            if preset is not None:
                return preset

    # 3) Only try YAML loading if it *looks like* a path / URL
    if _looks_like_preset_location(preset_og):
        try:
            loaded = load_preset_dict_from_location(preset_og)
        except Exception as e:
            raise ValueError(f"Failed to load preset from location {preset_og!r}: {e}") from e

        logger.log(20, f"Loaded presets from {preset_og!r}: keys={list(loaded.keys())}")
        return loaded

    # 4) Otherwise: ORIGINAL error behavior
    valid_presets = list(preset_dict.keys())

    raise ValueError(f"Preset '{preset_og}' was not found. Valid presets: {sorted(set(valid_presets))}")


def _apply_presets(preset_dict: Dict[str, dict], presets_alias: Dict[str, str] = None, *args, **kwargs):
    """
    Pair with `unpack` to alter input arguments with preset values.

    Parameters
    ----------
    preset_dict : Dict[str, dict]
        Dictionary of preset keys that map to dictionaries of key-word values.
    presets_alias : Dict[str, str], optional
        Dictionary of aliases of the presets in preset_dict.
        Aliases will be remapped to the original preset in preset_dict.
    presets : str or list, optional
        List of preset keys (and/or aliases) to apply.
        If str, then it is converted to a 1 element long list.
        presets are applied from first-to-last.
        If a key-word is specified in multiple presets in the list, the value will be set to the value of the last preset with that key-word.
    *args, **kwargs:
        The original args and kwargs (including presets as a potential kwarg).
        args and kwargs take priority over presets, and if specified in the input will not be overwritten.
        Presets will add new key-values to kwargs if the key did not previously exist.

    Returns
    -------
    (*args, **kwargs) with kwargs updated based on specified presets.
    """
    presets = kwargs.get("presets", None)
    if presets is None:
        return args, kwargs

    if not isinstance(presets, list):
        presets = [presets]

    preset_kwargs = {}
    for preset in presets:
        if isinstance(preset, str):
            preset_dict_resolved = _resolve_preset_str(preset, preset_dict, presets_alias)
            preset = preset_dict_resolved

        if isinstance(preset, dict):
            for key, val in preset.items():
                preset_kwargs[key] = val
        else:
            raise TypeError(
                f"Preset of type {type(preset)} was given, but only presets of type [dict, str] are valid."
            )

    # args/kwargs win over presets
    for key, val in preset_kwargs.items():
        if key not in kwargs:
            kwargs[key] = val

    return args, kwargs


def apply_presets(preset_dict: Dict[str, dict], presets_alias: Dict[str, str] = None):
    """Used as a decorator"""
    return unpack(_apply_presets, preset_dict, presets_alias)
