# TODO: Remove this file for v1.0 as space is moved to common
import sys
import warnings
from typing import Any

this_module = sys.modules[__name__]
spaces = ["Space", "Categorical", "Real", "Int", "Bool"]


class DeprecatedSpaceWrapper:
    def __init__(self, common_space) -> None:
        self.comomn_space = common_space

    def __call__(self, *args, **kwargs) -> Any:
        from autogluon.common import space

        warnings.warn(
            "Accessing search spaces through `autogluon.core` is deprecated as of v0.8 and won't be supported "
            f"in the next release. Please use `from autogluon.common import {self.comomn_space}` instead."
        )
        return getattr(space, self.comomn_space)(*args, **kwargs)


for s in spaces:
    setattr(this_module, s, DeprecatedSpaceWrapper(s))
