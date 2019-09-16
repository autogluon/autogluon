import pandas as pd

from tabular.sandbox.swallows.processors.constants import *
from tabular.sandbox.swallows.processors.preprocessor import AbstractStagePreprocessor
from tabular.sandbox.swallows.processors.stage_6_features_processor import Stage6FeaturesGenerator


class PreprocessorStage6(AbstractStagePreprocessor):

    def __init__(self, path, dataset_name, is_training, file_prefix_source, file_prefix_target, encoder_pickle_prefix=None):
        super(PreprocessorStage6, self).__init__(path, dataset_name, is_training, file_prefix_source, file_prefix_target)
        self.name = "PreprocessorStage6"
        self.encoder_pickle_prefix = encoder_pickle_prefix

    def load_stage_data(self):
        fpath = self.path / f'{self.file_prefix_source}.csv'
        df = pd.read_csv(fpath, low_memory=False, dtype=SOURCE_TYPE, encoding='utf-8')
        print(f'Stage data is loaded << {self.dataset_name} | {DATASETS[self.dataset_name]} | {fpath}')
        return df

    def get_processors(self):
        return [
            Stage6FeaturesGenerator(self.path, self.is_training, self.dataset_name, self.encoder_pickle_prefix),
        ]

    @classmethod
    def run_dataset(cls, path, dataset_name, encoder_pickle_prefix=None):
        PreprocessorStage6(path, dataset_name, DATASETS[dataset_name]['is_training'],
                           f'{dataset_name}',
                           f'processed/{dataset_name}_stage_6',
                           encoder_pickle_prefix
                           ).run()
