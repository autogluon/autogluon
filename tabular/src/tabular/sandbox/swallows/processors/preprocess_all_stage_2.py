from tabular.sandbox.swallows.processors.constants import DATASETS
from tabular.sandbox.swallows.processors.preprocessor import AbstractStagePreprocessor
from tabular.sandbox.swallows.processors.stage_2_amazon_links_processor import AmazonLinkProcessor
from tabular.sandbox.swallows.processors.stage_2_features_processor import Stage2FeaturesProcessor


class PreprocessorStage2(AbstractStagePreprocessor):

    def __init__(self, path, dataset_name, is_training, file_prefix_source, file_prefix_target, domains_pickle_prefix):
        super(PreprocessorStage2, self).__init__(path, dataset_name, is_training, file_prefix_source, file_prefix_target)
        self.domains_pickle_prefix = domains_pickle_prefix
        self.name = "PreprocessorStage2"

    def get_processors(self):
        return [
            Stage2FeaturesProcessor(self.is_training),

            # min_link_freq: 0>500 | 1=370 | 5=225 | 20=130 | 50=99
            AmazonLinkProcessor(self.path, self.is_training, self.dataset_name, 50, self.domains_pickle_prefix),
        ]

    @classmethod
    def run_dataset(cls, path, dataset_name, domains_pickle_prefix=None):
        PreprocessorStage2(path, dataset_name, DATASETS[dataset_name]['is_training'],
                           f'processed/{dataset_name}_stage_1',
                           f'processed/{dataset_name}_stage_2',
                           domains_pickle_prefix
                           ).run()
