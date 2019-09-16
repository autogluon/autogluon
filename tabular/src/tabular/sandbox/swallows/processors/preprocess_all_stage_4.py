from tabular.sandbox.swallows.processors.constants import *
from tabular.sandbox.swallows.processors.normalize_whitespace_processor import NormalizeWhitespaceProcessor
from tabular.sandbox.swallows.processors.preprocessor import AbstractStagePreprocessor
from tabular.sandbox.swallows.processors.stage_4_features_processor import Stage4Features


class PreprocessorStage4(AbstractStagePreprocessor):

    def __init__(self, path, dataset_name, is_training, file_prefix_source, file_prefix_target):
        super(PreprocessorStage4, self).__init__(path, dataset_name, is_training, file_prefix_source, file_prefix_target)
        self.name = "PreprocessorStage4"

    def get_processors(self):
        return [
            Stage4Features(),
            NormalizeWhitespaceProcessor('short_description')
        ]

    @classmethod
    def run_dataset(cls, path, dataset_name):
        PreprocessorStage4(path, dataset_name, DATASETS[dataset_name]['is_training'],
                           f'processed/{dataset_name}_stage_3',
                           f'processed/{dataset_name}_stage_4',
                           ).run()
