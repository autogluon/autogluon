import pandas as pd

from autogluon.eda import AnalysisState
from autogluon.eda.visualization.base import AbstractVisualization
from autogluon.eda.visualization.jupyter import JupyterMixin


class DatasetStatistics(AbstractVisualization, JupyterMixin):

    def __init__(self, headers: bool = False, namespace: str = None, **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers

    def can_handle(self, state: AnalysisState) -> bool:
        for k in ['dataset_stats', 'missing_statistics', 'raw_types']:
            if k in state:
                return True
        return False

    def _render(self, state: AnalysisState) -> None:
        # TODO: Is the namespace sample?
        sample_size = state.get('sample_size', None)

        datasets = []
        for k in ['dataset_stats', 'missing_statistics', 'raw_types']:
            if k in state:
                datasets = state[k].keys()

        for ds in datasets:
            # Merge different metrics
            stats = {}
            if 'dataset_stats' in state:
                stats = {**stats, **state.dataset_stats[ds]}
            if 'missing_statistics' in state:
                stats = {**stats, **{f'missing_{k}': v for k, v in state.missing_statistics[ds].items()}}
            if 'raw_types' in state:
                stats['raw_types'] = state.raw_types[ds]
            if 'special_types' in state:
                stats['special_types'] = state.special_types[ds]

            # Fix counts
            df = pd.DataFrame(stats)
            if 'dataset_stats' in state:
                for k in ['unique', 'freq']:
                    df[k] = df[k].fillna(-1).astype(int)
                df = df.fillna('')
                for k in ['unique', 'freq']:
                    df[k] = df[k].replace({-1: ''})

            df = df.fillna('')

            if self.headers:
                sample_info = '' if sample_size is None else f' (sample size: {sample_size})'
                header = f'{ds} dataset summary{sample_info}'
                self.render_text(header, text_type='h3')

            self.display_obj(df)
