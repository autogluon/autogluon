import copy
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# TODO: Rename to FeatureMetadata
class FeatureTypesMetadata:
    """
    Contains feature type metadata information such as type family groups (type_group_map_raw) and special feature type groups (type_group_map_special)

    # TODO: rewrite
    type_group_map_raw is the dictionary computed as output to :function:`autogluon.utils.tabular.features.utils.get_type_group_map_raw`
    type_group_map_special is an optional dictionary to communicate special properties of features to downstream models that have special handling functionality for those feature types.
        As an example, type_group_map_special might contain a key 'text_ngram' indicating that the list of values are all features which were generated from a nlp vectorizer and represent ngrams.
        A downstream model such as a K-Nearest-Neighbor model could then check if 'text_ngram' is present in type_group_map_special and drop those features if present, to speed up training and inference time.
    """
    def __init__(self, type_map_raw: dict, type_group_map_special: dict = None):
        if type_group_map_special is None:
            type_group_map_special = defaultdict(list)
        if not isinstance(type_group_map_special, defaultdict):
            type_group_map_special = defaultdict(list, type_group_map_special)

        self.type_map_raw = type_map_raw
        self.type_group_map_raw = self.get_type_group_map_raw_from_flattened(type_map_raw=self.type_map_raw)  # TODO: Move to after validate
        self.type_group_map_special = type_group_map_special

        self._validate()

    # Confirms if inputs are valid
    def _validate(self):
        type_group_map_raw_expanded = []
        for key in self.type_group_map_raw:
            type_group_map_raw_expanded += self.type_group_map_raw[key]
        type_group_map_special_expanded = []
        for key in self.type_group_map_special:
            type_group_map_special_expanded += self.type_group_map_special[key]
        type_group_map_raw_expanded_set = set(type_group_map_raw_expanded)
        if len(type_group_map_raw_expanded) != len(type_group_map_raw_expanded_set):
            raise AssertionError('type_group_map_raw contains features that appear multiple times!')

        for feature in type_group_map_special_expanded:
            if feature not in type_group_map_raw_expanded:
                raise AssertionError(f"feature '{feature}' is present in type_group_map_special but not in type_group_map_raw!")

    def get_features(self):
        return list(self.type_map_raw.keys())

    def get_feature_type_raw(self, feature):
        return self._get_feature_type(feature=feature, feature_types_dict=self.type_group_map_raw)

    def get_feature_type_special(self, feature):
        return self._get_feature_type(feature=feature, feature_types_dict=self.type_group_map_special)

    def get_type_group_map_raw_flattened(self):
        return {feature: type_family for type_family, features in self.type_group_map_raw.items() for feature in features}

    @staticmethod
    def get_type_group_map_raw_from_flattened(type_map_raw):
        type_group_map_raw = defaultdict(list)
        for feature, dtype in type_map_raw.items():
            type_group_map_raw[dtype].append(feature)
        return type_group_map_raw

    def remove_features(self, features, inplace=False):
        if inplace:
            metadata = self
        else:
            metadata = copy.deepcopy(self)
        metadata._remove_features_from_type_map(d=metadata.type_map_raw, features=features)
        metadata._remove_features_from_type_group_map(d=metadata.type_group_map_raw, features=features)
        metadata._remove_features_from_type_group_map(d=metadata.type_group_map_special, features=features)
        return metadata

    @staticmethod
    def _remove_features_from_type_group_map(d, features):
        for key, features_orig in d.items():
            d[key] = [feature for feature in features_orig if feature not in features]

    @staticmethod
    def _remove_features_from_type_map(d, features):
        for feature in features:
            if feature in d:
                d.pop(feature)

    # Joins two metadata objects together, returning a new metadata object
    def join_metadata(self, metadata, allow_shared_raw_features=False):
        type_map_raw = copy.deepcopy(self.type_map_raw)
        shared_features = []
        shared_features_diff_types = []
        for key, features in metadata.type_map_raw.items():
            if key in type_map_raw:
                shared_features.append(key)
                if type_map_raw[key] != metadata.type_map_raw[key]:
                    shared_features_diff_types.append(key)
        if not allow_shared_raw_features and shared_features:
            raise AssertionError(f'Metadata objects to join share a raw feature, but `allow_shared_raw_features=False`. Shared features: {shared_features}')
        if shared_features_diff_types:
            raise AssertionError(f'Metadata objects to join share raw features but do not agree on raw dtypes. Shared conflicting features: {shared_features_diff_types}')
        type_map_raw.update({key: val for key, val in metadata.type_map_raw.items() if key not in shared_features})

        type_group_map_special = copy.deepcopy(self.type_group_map_special)
        for key, features in metadata.type_group_map_special.items():
            if key in type_group_map_special:
                features_to_add = [feature for feature in features if feature not in type_group_map_special[key]]
                type_group_map_special[key] += features_to_add
            else:
                type_group_map_special[key] = features

        return FeatureTypesMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

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
