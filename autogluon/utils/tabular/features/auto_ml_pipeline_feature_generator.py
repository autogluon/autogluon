import logging

from .abstract_pipeline_feature_generator import AbstractPipelineFeatureGenerator
from .generators.category import CategoryFeatureGenerator
from .generators.text_special import TextSpecialFeatureGenerator
from .generators.identity import IdentityFeatureGenerator
from .generators.datetime import DatetimeFeatureGenerator
from .generators.text_ngram import TextNgramFeatureGenerator

logger = logging.getLogger(__name__)


class AutoMLPipelineFeatureGenerator(AbstractPipelineFeatureGenerator):
    def __init__(self, generators=None, enable_text_ngram_features=True, enable_text_special_features=True,
                 enable_categorical_features=True, enable_raw_features=True, enable_datetime_features=True,
                 vectorizer=None):
        self.enable_nlp_features = enable_text_ngram_features
        self.enable_text_special_features = enable_text_special_features
        self.enable_categorical_features = enable_categorical_features
        self.enable_raw_features = enable_raw_features
        self.enable_datetime_features = enable_datetime_features
        if generators is None:
            generators = self._get_default_generators(vectorizer=vectorizer)
        # TODO: no nlp preset must be addressed here! Currently doesn't work anymore!
        # TODO: Add generators_extra, generators_banned args.
        super().__init__(generators=generators)

    def _get_default_generators(self, vectorizer=None):
        generators = [
            IdentityFeatureGenerator(),
            CategoryFeatureGenerator(),
            DatetimeFeatureGenerator(),
            TextSpecialFeatureGenerator(),
            TextNgramFeatureGenerator(vectorizer=vectorizer),
        ]
        return generators
