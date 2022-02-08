from typing import Union, Optional, List, Tuple


def get_default_config_value(
        config: dict,
        keys: Tuple[str, ...],
):
    result = []
    for k, v in config.items():
        if k in keys:
            result.append(v)
        elif isinstance(v, dict):
            result += get_default_config_value(v, keys)
        else:
            pass

    return result
