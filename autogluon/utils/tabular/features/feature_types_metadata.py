import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class FeatureTypesMetadata:
    """
    Contains feature type metadata information such as type family groups (feature_types_raw) and special feature type groups (feature_types_special)

    feature_types_raw is the dictionary computed as output to :function:`autogluon.utils.tabular.features.utils.get_type_family_groups_df`
    feature_types_special is an optional dictionary to communicate special properties of features to downstream models that have special handling functionality for those feature types.
        As an example, feature_types_special might contain a key 'text_ngram' indicating that the list of values are all features which were generated from a nlp vectorizer and represent ngrams.
        A downstream model such as a K-Nearest-Neighbor model could then check if 'text_ngram' is present in feature_types_special and drop those features if present, to speed up training and inference time.
    """
    def __init__(self, feature_types_raw: defaultdict, feature_types_special: defaultdict = None):
        # These dictionaries must have only 1 instance of a feature among all of its value lists.
        if feature_types_special is None:
            feature_types_special = defaultdict(list)

        self.feature_types_raw = feature_types_raw
        self.feature_types_special = feature_types_special

        self._validate()

    # Confirms if inputs are valid
    def _validate(self):
        feature_types_raw_expanded = []
        for key in self.feature_types_raw:
            feature_types_raw_expanded += self.feature_types_raw[key]
        feature_types_special_expanded = []
        for key in self.feature_types_special:
            feature_types_special_expanded += self.feature_types_special[key]
        feature_types_raw_expanded_set = set(feature_types_raw_expanded)
        if len(feature_types_raw_expanded) != len(feature_types_raw_expanded_set):
            raise AssertionError('feature_types_raw contains features that appear multiple times!')

        for feature in feature_types_special_expanded:
            if feature not in feature_types_raw_expanded:
                raise AssertionError(f"feature '{feature}' is present in feature_types_special but not in feature_types_raw!")

    def get_feature_type_raw(self, feature):
        return self._get_feature_type(feature=feature, feature_types_dict=self.feature_types_raw)

    def get_feature_type_special(self, feature):
        return self._get_feature_type(feature=feature, feature_types_dict=self.feature_types_special)

    def get_feature_types_raw_flattened(self):
        return {feature: type_family for type_family, features in self.feature_types_raw.items() for feature in features}

    @staticmethod
    def _get_feature_type(feature, feature_types_dict):
        for dtype_family in feature_types_dict:
            if feature in feature_types_dict[dtype_family]:
                return dtype_family
            else:
                raise ValueError(f'Feature {feature} not found in provided feature_types_dict!')
