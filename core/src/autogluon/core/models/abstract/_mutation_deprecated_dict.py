"""Dict containers that deprecate post-construction mutation.

A model's resolved configuration (`params`, `params_aux`) must not be mutated after
construction: values computed at runtime belong on the model instance (or in
`params_trained` for fit-resolved hyperparameters), and user configuration flows in
through the constructor. :class:`MutationDeprecatedDict` enforces this as a deprecation:
mutating operations emit a DeprecationWarning now and raise a TypeError starting in
AutoGluon ``1.7`` (matching the eventual read-only mapping behavior). ``copy()`` returns
a plain (mutable) dict, so copy-and-edit code is unaffected.
"""

from __future__ import annotations

import warnings

from packaging import version

from ...version import __version__

_MUTATION_RAISE_MIN_VERSION = "1.7"
"""AutoGluon version from which mutating these containers raises instead of warning."""

_MUTATION_SHOULD_RAISE = version.parse(__version__) >= version.parse(_MUTATION_RAISE_MIN_VERSION)


class MutationDeprecatedDict(dict):
    """dict whose mutating operations are deprecated; see the module docstring.

    Subclasses set `_MUTATION_MSG` to a message naming the container and the
    replacement patterns.
    """

    _MUTATION_MSG: str = "Mutating this container after construction is deprecated."

    @classmethod
    def _mutation_deprecated(cls):
        if _MUTATION_SHOULD_RAISE:
            raise TypeError(cls._MUTATION_MSG)
        warnings.warn(cls._MUTATION_MSG, category=DeprecationWarning, stacklevel=3)

    def __reduce__(self):
        # Reconstruct through the constructor: pickle's default protocol for dict
        # subclasses repopulates item-by-item via `__setitem__`, which would trip the
        # deprecation on every load/deepcopy.
        return (self.__class__, (dict(self),))

    def __setitem__(self, key, value):
        self._mutation_deprecated()
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self._mutation_deprecated()
        super().__delitem__(key)

    def __ior__(self, other):
        self._mutation_deprecated()
        return super().__ior__(other)

    def update(self, *args, **kwargs):
        self._mutation_deprecated()
        super().update(*args, **kwargs)

    def setdefault(self, key, default=None):
        if key not in self:  # only an actual insertion is a mutation
            self._mutation_deprecated()
        return super().setdefault(key, default)

    def pop(self, *args, **kwargs):
        self._mutation_deprecated()
        return super().pop(*args, **kwargs)

    def popitem(self):
        self._mutation_deprecated()
        return super().popitem()

    def clear(self):
        self._mutation_deprecated()
        super().clear()


class ParamsDict(MutationDeprecatedDict):
    """`params` (model hyperparameters) container; deprecates post-construction mutation."""

    _MUTATION_MSG = (
        "Mutating a model's `params` after construction is deprecated and will raise an "
        f"exception starting in AutoGluon {_MUTATION_RAISE_MIN_VERSION}. `params` is the model's "
        "resolved hyperparameter configuration: work on the copy returned by `_get_model_params()`, "
        "record hyperparameter values resolved during fit in `params_trained`, and store other "
        "runtime values as instance state. User configuration flows in via the `hyperparameters` "
        "constructor argument."
    )
