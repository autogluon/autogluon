import functools
import logging

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


def _apply_presets(preset_dict, *args, **kwargs):
    """
    Pair with `unpack` to alter input arguments with preset values.
    """
    if 'presets' in kwargs:
        presets = kwargs['presets']
        if presets is None:
            return kwargs
        if not isinstance(presets, list):
            presets = [presets]
        preset_kwargs = {}
        for preset in presets:
            if isinstance(preset, str):
                preset_orig = preset
                preset = preset_dict.get(preset, None)
                if preset is None:
                    raise ValueError(f'Preset \'{preset_orig}\' was not found. Valid presets: {list(preset_dict.keys())}')
            if isinstance(preset, dict):
                for key in preset:
                    preset_kwargs[key] = preset[key]
            else:
                raise TypeError(f'Preset of type {type(preset)} was given, but only presets of type [dict, str] are valid.')
        for key in preset_kwargs:
            if key not in kwargs:
                kwargs[key] = preset_kwargs[key]
    return args, kwargs


def apply_presets(preset_dict):
    """Used as a decorator"""
    return unpack(_apply_presets, preset_dict)
