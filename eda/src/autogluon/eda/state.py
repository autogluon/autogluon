import logging
from typing import List

logger = logging.getLogger(__name__)


class AnalysisState(dict):
    """Enabling dot.notation access to dictionary attributes and dynamic code assist in jupyter"""
    __getattr__ = dict.get
    __delattr__ = dict.__delitem__

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
    def at_least_one_key_must_be_present(self, state: AnalysisState, keys: List[str]):
        """
        Checks if at least one key is present in the state

        Parameters
        ----------
        state: AnalysisState
            state object to perform check on
        keys: List[str]
            list of the keys to check

        Returns
        -------
            True if at least one key from the `keys` list is present in the state
        """
        for k in keys:
            if k in state:
                return True
        logger.warning(f'{self.__class__.__name__}: at least one of the following keys must be present: {keys}')
        return False

    def all_keys_must_be_present(self, state: AnalysisState, keys: List[str]):
        """
        Checks if all the keys are present in the state

        Parameters
        ----------
        state: AnalysisState
            state object to perform check on
        keys: List[str]
            list of the keys to check

        Returns
        -------
            True if all the key from the `keys` list are present in the state
        """
        keys_not_present = [k for k in keys if k not in state.keys()]
        can_handle = len(keys_not_present) == 0
        if not can_handle:
            logger.warning(f'{self.__class__.__name__}: all of the following keys must be present: {keys}. The following keys are missing: {keys_not_present}')
        return can_handle
