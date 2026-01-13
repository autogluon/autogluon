import logging

__all__ = ["AnalysisState", "StateCheckMixin", "is_key_present_in_state"]

from typing import Any


class AnalysisState(dict):
    """Enabling dot.notation access to dictionary attributes and dynamic code assist in jupyter"""

    _getattr__ = dict.get
    __delattr__ = dict.__delitem__  # type: ignore

    def __getattr__(self, item) -> Any:  # needed for mypy checks
        return self._getattr__(item)

    def __init__(self, *args, **kwargs) -> None:
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        for k, v in kwargs.items():
            self[k] = v

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, dict):
            value = AnalysisState(value)
        self[name] = value

    def __setitem__(self, key, value) -> None:
        if isinstance(value, dict):
            value = AnalysisState(value)
        super().__setitem__(key, value)

    @property
    def __dict__(self):
        return self


class StateCheckMixin:
    logger = logging.getLogger(__name__)

    def at_least_one_key_must_be_present(self, state: AnalysisState, *keys) -> bool:
        """
        Checks if at least one key is present in the state

        Parameters
        ----------
        state: AnalysisState
            state object to perform check on
        keys:
            list of the keys to check

        Returns
        -------
            True if at least one key from the `keys` list is present in the state
        """
        for k in keys:
            if state.get(k, None) is not None:
                return True
        self.logger.warning(f"{self.__class__.__name__}: at least one of the following keys must be present: {keys}")
        return False

    def all_keys_must_be_present(self, state: AnalysisState, *keys) -> bool:
        """
        Checks if all the keys are present in the state

        Parameters
        ----------
        state: AnalysisState
            state object to perform check on
        keys:
            list of the keys to check

        Returns
        -------
            True if all the key from the `keys` list are present in the state
        """
        keys_not_present = [k for k in keys if state.get(k, None) is None]
        can_handle = len(keys_not_present) == 0
        if not can_handle:
            self.logger.warning(
                f"{self.__class__.__name__}: all of the following keys must be present: [{', '.join(keys)}]. "
                f"The following keys are missing: [{', '.join(keys_not_present)}]"
            )
        return can_handle


def is_key_present_in_state(state: AnalysisState, key: str):
    """
    Check if the nested key represented with dot notation (`a.b.c`) is present in the state
    Parameters
    ----------
    state: AnalysisState
        state to check the key in
    key: str
        the key to check for presence


    Returns
    -------
    `True` if the key is present

    """
    path = state
    for p in key.split("."):
        if p not in path:
            return False
        path = path[p]
    return True
