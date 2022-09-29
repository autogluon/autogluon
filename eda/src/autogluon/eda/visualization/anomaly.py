from matplotlib import pyplot as plt
from .base import AbstractVisualization
from .jupyter import JupyterMixin
from .. import AnalysisState
from shap.plots import waterfall


class AnomalyVisualization(AbstractVisualization, JupyterMixin):
    """
    Return the top k anomalies and their Shapely values, via a waterfall plot.
    """

    def __init__(self, headers: bool = False, namespace: str = None, **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers

    def can_handle(self, state: AnalysisState) -> bool:
        keys_pres = self._all_keys_must_be_present(state, ['top_anomalies'])
        enough_test = state.test_ano_scores.shape[0] >= len(state.top_anomalies)
        return keys_pres and enough_test

    def _display_anom_water(self, test_samp, shap_data) -> None:
        plt.clf()
        fig = waterfall(shap_data, show=False)
        self.display_obj(test_samp.to_frame().T)
        self.display_obj(fig)

    def _render(self, state: AnalysisState) -> None:
        header_text = f'Top {len(state.top_anomalies)} anomalies in test set'
        self.render_header_if_needed(state, header_text)
        for test_samp, shap_data in state.top_anomalies:
            self._display_anom_water(test_samp, shap_data)
