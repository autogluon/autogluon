from unittest.mock import MagicMock

from autogluon.eda import AnalysisState
from autogluon.eda.visualization.base import AbstractVisualization


class TestVizualization(AbstractVisualization):

    def can_handle(self, state: AnalysisState) -> bool:
        return 'required_key' in state

    def _render(self, state: AnalysisState) -> None:
        pass


def test_abstractvisualization_cannot_render():
    viz = TestVizualization()
    viz._render = MagicMock()
    viz.can_handle = MagicMock(wraps=viz.can_handle)
    viz.render(AnalysisState({'ns1': {'required_key': True, 'data': 1}, 'ns2': {}}))
    viz.can_handle.assert_called_once()
    viz._render.assert_not_called()


def test_abstractvisualization_can_render():
    viz = TestVizualization(namespace='ns1')
    viz._render = MagicMock()
    viz.can_handle = MagicMock(wraps=viz.can_handle)
    viz.render(AnalysisState({'ns1': {'required_key': True, 'data': 1}, 'ns2': {}}))
    viz.can_handle.assert_called_once()
    viz._render.assert_called_with({'required_key': True, 'data': 1})
