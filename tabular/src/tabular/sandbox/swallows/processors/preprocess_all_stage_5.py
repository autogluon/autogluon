from tabular.sandbox.swallows.processors.constants import *
from tabular.sandbox.swallows.processors.preprocessor import AbstractStagePreprocessor
from tabular.sandbox.swallows.processors.stage_5_features_processor import TargetMeanEncoder


class PreprocessorStage5(AbstractStagePreprocessor):

    def __init__(self, path, dataset_name, is_training, file_prefix_source, file_prefix_target, encoder_pickle_prefix=None):
        super(PreprocessorStage5, self).__init__(path, dataset_name, is_training, file_prefix_source, file_prefix_target)
        self.name = "PreprocessorStage5"
        self.encoder_pickle_prefix = encoder_pickle_prefix

    def get_processors(self):
        return [
            TargetMeanEncoder(self.path, self.is_training, self.dataset_name, self.encoder_pickle_prefix),
        ]

    @classmethod
    def run_dataset(cls, path, dataset_name, encoder_pickle_prefix=None):
        PreprocessorStage5(path, dataset_name, DATASETS[dataset_name]['is_training'],
                           f'processed/{dataset_name}_stage_4',
                           f'processed/{dataset_name}_stage_5',
                           encoder_pickle_prefix
                           ).run()
