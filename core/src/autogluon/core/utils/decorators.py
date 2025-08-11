import functools
import logging
from typing import Dict

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
    if presets is not None:
        if not isinstance(presets, list):
            presets = [presets]
        preset_kwargs = {}
        for preset in presets:
            if isinstance(preset, str):
                preset_og = preset
                preset = preset_dict.get(preset_og, None)
                if preset is None and presets_alias is not None:
                    preset = presets_alias.get(preset_og, None)
                    if preset is not None:
                        logger.log(20, f"Preset alias specified: '{preset_og}' maps to '{preset}'.")
                        preset = preset_dict.get(preset, None)
                if preset is None:
                    raise ValueError(f"Preset '{preset_og}' was not found. Valid presets: {list(preset_dict.keys())}")
            if isinstance(preset, dict):
                for key in preset:
                    preset_kwargs[key] = preset[key]
            else:
                raise TypeError(
                    f"Preset of type {type(preset)} was given, but only presets of type [dict, str] are valid."
                )
        for key in preset_kwargs:
            if key not in kwargs:
                kwargs[key] = preset_kwargs[key]
    return args, kwargs


def apply_presets(preset_dict: Dict[str, dict], presets_alias: Dict[str, str] = None):
    """Used as a decorator"""
    return unpack(_apply_presets, preset_dict, presets_alias)
