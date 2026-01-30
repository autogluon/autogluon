from __future__ import annotations

from typing import TypeAlias

from autogluon.common import space

SearchableInt: TypeAlias = int | space.Int
SearchableBool: TypeAlias = bool | space.Bool
SearchableFloat: TypeAlias = float | space.Real
