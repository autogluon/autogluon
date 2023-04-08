from typing import Any, Dict

__all__ = ["get_empty_dict_if_none", "expand_nested_args_into_nested_maps"]


def get_empty_dict_if_none(value) -> dict:
    if value is None:
        value = {}
    return value


def expand_nested_args_into_nested_maps(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expands flat args with nested keys in dot notation into nested map (`{a.b.c: value} -> {'a': {'b': {'c': value}}}`)

    Parameters
    ----------
    args: Dict[str, Any]
        args to expand

    Returns
    -------
    nested expanded map

    """
    result: Dict[str, Any] = {}
    for k, v in args.items():
        sub_keys = k.split(".")
        curr_pointer = result
        if len(sub_keys) > 1:
            for subkey in sub_keys[:-1]:
                if subkey not in curr_pointer:
                    curr_pointer[subkey] = {}
                curr_pointer = curr_pointer[subkey]
        if type(curr_pointer) is not dict:
            raise ValueError(f"{k} cannot be added - the key is already present")
        curr_pointer[sub_keys[-1]] = v
    return result
