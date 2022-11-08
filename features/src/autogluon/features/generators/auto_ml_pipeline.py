import logging

from autogluon.common.features.types import R_INT, R_FLOAT, S_TEXT, R_OBJECT, S_IMAGE_PATH

from .pipeline import PipelineFeatureGenerator
from .category import CategoryFeatureGenerator
from .datetime import DatetimeFeatureGenerator
from .identity import IdentityFeatureGenerator
from .isnan import IsNanFeatureGenerator
from .text_ngram import TextNgramFeatureGenerator
from .text_special import TextSpecialFeatureGenerator

logger = logging.getLogger(__name__)


# TODO: write out in English the full set of transformations that are applied (and eventually host page on website).
#  Also explicitly write out all of the feature-generator "hyperparameters" that might affect the results from the AutoML FeatureGenerator
class AutoMLPipelineFeatureGenerator(PipelineFeatureGenerator):
    """
    Pipeline feature generator with simplified arguments to handle most Tabular data including text and dates adequately.
    This is the default feature generation pipeline used by AutoGluon when unspecified.
    For more customization options, refer to :class:`PipelineFeatureGenerator` and :class:`BulkFeatureGenerator`.

    Parameters
    ----------
    enable_numeric_features : bool, default True
        Whether to keep features of 'int' and 'float' raw types.
        These features are passed without alteration to the models.
        Appends IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=['int', 'float']))) to the generator group.
    enable_categorical_features : bool, default True
        Whether to keep features of 'object' and 'category' raw types.
        These features are processed into memory optimized 'category' features.
        Appends CategoryFeatureGenerator() to the generator group.
    enable_datetime_features : bool, default True
        Whether to keep features of 'datetime' raw type and 'object' features identified as 'datetime_as_object' features.
        These features will be converted to 'int' features representing milliseconds since epoch.
        Appends DatetimeFeatureGenerator() to the generator group.
    enable_text_special_features : bool, default True
        Whether to use 'object' features identified as 'text' features to generate 'text_special' features such as word count,
        capital letter ratio, and symbol counts.
        Appends TextSpecialFeatureGenerator() to the generator group.
    enable_text_ngram_features : bool, default True
        Whether to use 'object' features identified as 'text' features to generate 'text_ngram' features.
        Appends TextNgramFeatureGenerator(vectorizer=vectorizer, text_ngram_params) to the generator group. See text_ngram.py for valid parameters.
    enable_raw_text_features : bool, default False
        Whether to use the raw text features. The generated raw text features will end up with '_raw_text' suffix.
        For example, 'sentence' --> 'sentence_raw_text'
    enable_vision_features : bool, default True
        [Experimental]
        Whether to keep 'object' features identified as 'image_path' special type.
        Features of this form should have a string path to an image file as their value.
        Only vision models can leverage these features, and these features will not be treated as categorical.
        Note: 'image_path' features will not be automatically inferred. These features must be explicitly specified as such in a custom FeatureMetadata object.
        Note: It is recommended that the string paths use absolute paths rather than relative, as it will likely be more stable.
    vectorizer : :class:`sklearn.feature_extraction.text.CountVectorizer`, default CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=10000, dtype=np.uint8)  # noqa
        sklearn CountVectorizer object to use in :class:`TextNgramFeatureGenerator`.
        Only used if `enable_text_ngram_features=True`.
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.

    Examples
    --------
    >>> from autogluon.tabular import TabularDataset
    >>> from autogluon.features.generators import AutoMLPipelineFeatureGenerator
    >>>
    >>> feature_generator = AutoMLPipelineFeatureGenerator()
    >>>
    >>> label = 'class'
    >>> train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    >>> X_train = train_data.drop(labels=[label], axis=1)
    >>> y_train = train_data[label]
    >>>
    >>> X_train_transformed = feature_generator.fit_transform(X=X_train, y=y_train)
    >>>
    >>> test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    >>>
    >>> X_test_transformed = feature_generator.transform(test_data)
    """
    def __init__(self,
                 enable_numeric_features=True,
                 enable_categorical_features=True,
                 enable_datetime_features=True,
                 enable_text_special_features=True,
                 enable_text_ngram_features=True,
                 enable_raw_text_features=False,
                 enable_vision_features=True,
                 vectorizer=None,
                 text_ngram_params=None,
                 **kwargs):
        if 'generators' in kwargs:
            raise KeyError(f'generators is not a valid parameter to {self.__class__.__name__}. '
                           f'Use {PipelineFeatureGenerator.__name__} to specify custom generators.')
        if 'enable_raw_features' in kwargs:
            enable_numeric_features = kwargs.pop('enable_raw_features')
            logger.warning("'enable_raw_features is a deprecated parameter, use 'enable_numeric_features' instead. "
                           "Specifying 'enable_raw_features' will raise an exception starting in 0.1.0")

        self.enable_numeric_features = enable_numeric_features
        self.enable_categorical_features = enable_categorical_features
        self.enable_datetime_features = enable_datetime_features
        self.enable_text_special_features = enable_text_special_features
        self.enable_text_ngram_features = enable_text_ngram_features
        self.enable_raw_text_features = enable_raw_text_features
        self.enable_vision_features = enable_vision_features
        self.text_ngram_params = text_ngram_params if text_ngram_params else {}

        generators = self._get_default_generators(vectorizer=vectorizer)
        super().__init__(generators=generators, **kwargs)

    def _get_default_generators(self, vectorizer=None):
        generator_group = []
        if self.enable_numeric_features:
            generator_group.append(IdentityFeatureGenerator(infer_features_in_args=dict(
                valid_raw_types=[R_INT, R_FLOAT])))
        if self.enable_raw_text_features:
            generator_group.append(IdentityFeatureGenerator(infer_features_in_args=dict(
                required_special_types=[S_TEXT], invalid_special_types=[S_IMAGE_PATH]), name_suffix='_raw_text'))
        if self.enable_categorical_features:
            generator_group.append(CategoryFeatureGenerator())
        if self.enable_datetime_features:
            generator_group.append(DatetimeFeatureGenerator())
        if self.enable_text_special_features:
            generator_group.append(TextSpecialFeatureGenerator())
        if self.enable_text_ngram_features:
            generator_group.append(TextNgramFeatureGenerator(vectorizer=vectorizer, **self.text_ngram_params))
        if self.enable_vision_features:
            generator_group.append(IdentityFeatureGenerator(infer_features_in_args=dict(
                valid_raw_types=[R_OBJECT], required_special_types=[S_IMAGE_PATH],
            )))
            generator_group.append(IsNanFeatureGenerator(infer_features_in_args=dict(
                valid_raw_types=[R_OBJECT], required_special_types=[S_IMAGE_PATH],
            )))
        generators = [generator_group]
        return generators
