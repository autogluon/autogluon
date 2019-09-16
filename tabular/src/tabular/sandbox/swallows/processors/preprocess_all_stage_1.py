import pandas as pd

from tabular.sandbox.swallows.processors.constants import SOURCE_TYPE, DATASETS, DATASET_SAMPLE
from tabular.sandbox.swallows.processors.normalize_whitespace_processor import NormalizeWhitespaceProcessor
from tabular.sandbox.swallows.processors.preprocessor import AbstractStagePreprocessor
from tabular.sandbox.swallows.processors.stage_1_add_language_processor import AddDetailsLanguage
from tabular.sandbox.swallows.processors.stage_1_unescape_processor import UnescapeProcessor


class PreprocessorStage1(AbstractStagePreprocessor):

    def __init__(self, path, dataset_name, is_training, file_prefix_source, file_prefix_target):
        super(PreprocessorStage1, self).__init__(path, dataset_name, is_training, file_prefix_source, file_prefix_target)
        self.name = "PreprocessorStage1"

    def get_processors(self):
        return [
            UnescapeProcessor(),
            NormalizeWhitespaceProcessor(),
            AddDetailsLanguage(),

        ]

    def load_stage_data(self):
        df = pd.read_csv(self.path / f'{self.file_prefix_source}.csv', low_memory=False, dtype=SOURCE_TYPE, encoding='utf-8')
        print(f'Stage data is loaded << {self.dataset_name} | {DATASETS[self.dataset_name]}')
        return df

    @classmethod
    def run_dataset(cls, path, dataset_name):
        PreprocessorStage1(path, dataset_name, DATASETS[dataset_name]['is_training'],
                           file_prefix_source=f'{dataset_name}',
                           file_prefix_target=f'processed/{dataset_name}_stage_1'
                           ).run()
