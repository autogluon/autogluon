import logging

from .pipeline import PipelineFeatureGenerator
from .category import CategoryFeatureGenerator
from .datetime import DatetimeFeatureGenerator
from .identity import IdentityFeatureGenerator
from .text_ngram import TextNgramFeatureGenerator
from .text_special import TextSpecialFeatureGenerator
from ..feature_metadata import R_INT, R_FLOAT

logger = logging.getLogger(__name__)


class AutoMLPipelineFeatureGenerator(PipelineFeatureGenerator):
    """
    Pipeline feature generator with simplified arguments to handle most Tabular data including text and dates adequately.
    This is the default feature generation pipeline used by AutoGluon when unspecified.
    For more customization options, refer to PipelineFeatureGenerator and BulkFeatureGenerator.

    Parameters
    ----------
    enable_raw_features : bool, default True
        Enables raw feature types to be kept.
        This is typically any feature which is not of the types ['object', 'category', 'datetime'].
        Appends IdentityFeatureGenerator() to the generator group.
    enable_categorical_features : bool, default True
        Enables 'object' and 'category' feature types to be kept and processed into memory optimized category features.
        Appends CategoryFeatureGenerator() to the generator group.
    enable_datetime_features : bool, default True
        Enables 'datetime' features and 'object' features identified as 'datetime_as_object' features to be processed as integers.
        Appends DatetimeFeatureGenerator() to the generator group.
    enable_text_special_features : bool, default True
        Enables 'object' features identified as 'text' features to generate 'text_special' features such as word count, capital letter ratio, and symbol counts.
        Appends TextSpecialFeatureGenerator() to the generator group.
    enable_text_ngram_features : bool, default True
        Enables 'object' features identified as 'text' features to generate 'text_ngram' features.
        Appends TextNgramFeatureGenerator(vectorizer=vectorizer) to the generator group.
    vectorizer : CountVectorizer, default None
        sklearn CountVectorizer object to use in TextNgramFeatureGenerator.
        If None, then the default CountVectorizer is used.

    >>> from autogluon import TabularPrediction as task
    >>> from autogluon.utils.tabular.features.generators import AutoMLPipelineFeatureGenerator
    >>>
    >>> feature_generator = AutoMLPipelineFeatureGenerator()
    >>>
    >>> label_column = 'class'
    >>> train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    >>> X_train = train_data.drop(labels=[label_column], axis=1)
    >>> y_train = train_data[label_column]
    >>>
    >>> X_train_transformed = feature_generator.fit_transform(X=X_train, y=y_train)
    >>>
    >>> test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    >>>
    >>> X_test_transformed = feature_generator.transform(test_data)
    """
    def __init__(self, enable_raw_features=True, enable_categorical_features=True, enable_datetime_features=True,
                 enable_text_special_features=True, enable_text_ngram_features=True, vectorizer=None, **kwargs):
        if 'generators' in kwargs:
            raise KeyError(f'generators is not a valid parameter to {self.__class__.__name__}. Use {PipelineFeatureGenerator.__name__} to specify custom generators.')

        self.enable_raw_features = enable_raw_features
        self.enable_categorical_features = enable_categorical_features
        self.enable_datetime_features = enable_datetime_features
        self.enable_text_special_features = enable_text_special_features
        self.enable_text_ngram_features = enable_text_ngram_features

        generators = self._get_default_generators(vectorizer=vectorizer)
        super().__init__(generators=generators, **kwargs)

    def _get_default_generators(self, vectorizer=None):
        generator_group = []
        if self.enable_raw_features:
            generator_group.append(IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])))
        if self.enable_categorical_features:
            generator_group.append(CategoryFeatureGenerator())
        if self.enable_datetime_features:
            generator_group.append(DatetimeFeatureGenerator())
        if self.enable_text_special_features:
            generator_group.append(TextSpecialFeatureGenerator())
        if self.enable_text_ngram_features:
            generator_group.append(TextNgramFeatureGenerator(vectorizer=vectorizer))
        generators = [generator_group]
        return generators
