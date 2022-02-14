from typing import Tuple


def get_default_config_value(
        config: dict,
        keys: Tuple[str, ...],
):
    """
    Traverse a config dictionary to get some default hyper-parameter.

    Parameters
    ----------
    config
        A config dictionary.
    keys
        The possible names of a hyper-parameter.

    Returns
    -------
    The hyper-parameter value.
    """
    result = []
    for k, v in config.items():
        if k in keys:
            result.append(v)
        elif isinstance(v, dict):
            result += get_default_config_value(v, keys)
        else:
            pass

    return result
