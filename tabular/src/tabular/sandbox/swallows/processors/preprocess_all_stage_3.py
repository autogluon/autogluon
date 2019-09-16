from tabular.sandbox.swallows.processors.constants import *
from tabular.sandbox.swallows.processors.normalize_whitespace_processor import NormalizeWhitespaceProcessor
from tabular.sandbox.swallows.processors.preprocessor import AbstractStagePreprocessor
from tabular.sandbox.swallows.processors.stage_3_common_text_removal_2_processor import Stage3CommonTextRemovalProcessor2
from tabular.sandbox.swallows.processors.stage_3_common_text_removal_3_processor import Stage3CommonTextRemovalProcessor3
from tabular.sandbox.swallows.processors.stage_3_common_text_removal_processor import Stage3CommonTextRemovalProcessor
from tabular.sandbox.swallows.processors.stage_3_date_features_processor import Stage3DateFeaturesProcessor
from tabular.sandbox.swallows.processors.stage_3_token_features_processor import Stage3TokenFeaturesProcessor


class PreprocessorStage3(AbstractStagePreprocessor):

    def __init__(self, path, dataset_name, is_training, file_prefix_source, file_prefix_target):
        super(PreprocessorStage3, self).__init__(path, dataset_name, is_training, file_prefix_source, file_prefix_target)
        self.name = "PreprocessorStage3"

    def get_processors(self):
        return [
            Stage3CommonTextRemovalProcessor(),
            Stage3TokenFeaturesProcessor(),
            NormalizeWhitespaceProcessor(),

            Stage3CommonTextRemovalProcessor2(),
            NormalizeWhitespaceProcessor(),

            Stage3CommonTextRemovalProcessor3(),
            NormalizeWhitespaceProcessor(),

            Stage3DateFeaturesProcessor(),
        ]

    @classmethod
    def run_dataset(cls, path, dataset_name):
        PreprocessorStage3(path, dataset_name, DATASETS[dataset_name]['is_training'],
                           f'processed/{dataset_name}_stage_2',
                           f'processed/{dataset_name}_stage_3',
                           ).run()
