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
        keys_pres = 'top_train_anomalies' in state or 'top_test_anomalies' in state
        return keys_pres

    def _display_anom_water(self, score, test_samp, shap_data) -> None:
        plt.clf()
        fig = waterfall(shap_data, show=False)
        test_samp['anomaly score'] = round(score, 3)
        cols = list(test_samp.index)
        test_samp = test_samp.reindex(index = [cols[-1]] + cols[:-1])
        self.display_obj(test_samp.to_frame().T)
        self.display_obj(fig)

    def _render(self, state: AnalysisState) -> None:
        if 'top_train_anomalies' in state:
            header_text = f'Top {len(state.top_train_anomalies)} anomalies in training set'
            self.render_header_if_needed(state, header_text)
            for score, test_samp, shap_data in state.top_train_anomalies:
                self._display_anom_water(score, test_samp, shap_data)
        if 'top_test_anomalies' in state:
            header_text = f'Top {len(state.top_test_anomalies)} anomalies in test set'
            self.render_header_if_needed(state, header_text)
            for score, test_samp, shap_data in state.top_test_anomalies:
                self._display_anom_water(score, test_samp, shap_data)
