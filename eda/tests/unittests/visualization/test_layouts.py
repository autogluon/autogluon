from unittest.mock import MagicMock

import pytest

from autogluon.eda import AnalysisState
from autogluon.eda.visualization import PropertyRendererComponent


@pytest.mark.parametrize("with_transform_fn, expected", [(True, "VALUE"), (False, "value")])
def test_PropertyRendererComponent(with_transform_fn, expected):
    state = AnalysisState({"some": {"prop": "value"}})
    call_display_obj = MagicMock()

    transform_fn = (lambda v: v.upper()) if with_transform_fn else None

    viz = PropertyRendererComponent("some.prop", transform_fn=transform_fn)
    viz.display_obj = call_display_obj

    viz.render(state)

    call_display_obj.assert_called_with(expected)
