import copy
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# TODO: Rename to FeatureMetadata
# TODO: input should be a dictionary of feature name -> raw type family / raw type, then construct feature_types_raw from that and have both representations.
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
        if not isinstance(feature_types_raw, defaultdict):
            feature_types_raw = defaultdict(list, feature_types_raw)
        if not isinstance(feature_types_special, defaultdict):
            feature_types_special = defaultdict(list, feature_types_special)

        # self.type_map_raw = self.get_feature_types_raw_flattened()
        # self.type_map_raw = type_map_raw
        self.feature_types_raw = feature_types_raw
        # self.feature_types_raw = self.get_feature_types_raw_from_flattened(flattened=self.type_map_raw)  # TODO: Move to after validate
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
    def get_feature_types_raw_from_flattened(flattened):
        feature_types_raw = defaultdict(list)
        for feature, dtype in flattened.items():
            feature_types_raw[dtype].append(feature)
        return feature_types_raw

    def remove_features(self, features, inplace=False):
        if inplace:
            metadata = self
        else:
            metadata = copy.deepcopy(self)
        metadata._remove_features_from_dict(d=metadata.feature_types_raw, features=features)
        metadata._remove_features_from_dict(d=metadata.feature_types_special, features=features)
        return metadata

    @staticmethod
    def _remove_features_from_dict(d, features):
        for key, features_orig in d.items():
            d[key] = [feature for feature in features_orig if feature not in features]

    # Joins two metadata objects together, returning a new metadata object
    def join_metadata(self, metadata, allow_shared_raw_features=False):
        feature_types_raw = copy.deepcopy(self.feature_types_raw)
        for key, features in metadata.feature_types_raw.items():
            if key in feature_types_raw:
                features_to_add = [feature for feature in features if feature not in feature_types_raw[key]]
                if not allow_shared_raw_features:
                    if features_to_add != features:
                        shared_raw_features = set(features) - set(features_to_add)
                        raise AssertionError(f'Metadata objects to join share a raw feature, but `allow_shared_raw_features=False`. Shared features: {shared_raw_features}')
                feature_types_raw[key] += features_to_add
            else:
                feature_types_raw[key] = features

        feature_types_special = copy.deepcopy(self.feature_types_special)
        for key, features in metadata.feature_types_special.items():
            if key in feature_types_special:
                features_to_add = [feature for feature in features if feature not in feature_types_special[key]]
                feature_types_special[key] += features_to_add
            else:
                feature_types_special[key] = features

        return FeatureTypesMetadata(feature_types_raw=feature_types_raw, feature_types_special=feature_types_special)

    @staticmethod
    def _get_feature_type(feature, feature_types_dict):
        for dtype_family in feature_types_dict:
            if feature in feature_types_dict[dtype_family]:
                return dtype_family
            else:
                raise ValueError(f'Feature {feature} not found in provided feature_types_dict!')

    # Joins a list of metadata objects together, returning a new metadata object
    @staticmethod
    def join_metadatas(metadata_list, allow_shared_raw_features=False):
        metadata_new = copy.deepcopy(metadata_list[0])
        for metadata in metadata_list[1:]:
            metadata_new = metadata_new.join_metadata(metadata, allow_shared_raw_features=allow_shared_raw_features)
        return metadata_new
