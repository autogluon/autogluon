import copy
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class FeatureMetadata:
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
            raise AssertionError('type_group_map_raw contains features that appear multiple times.')

        features_invalid = []
        for feature in type_group_map_special_expanded:
            if feature not in type_group_map_raw_expanded:
                features_invalid.append(feature)
        if features_invalid:
            raise AssertionError(f"{len(features_invalid)} features are present in type_group_map_special but not in type_group_map_raw. Invalid features: {features_invalid}")

    def get_features(self):
        return list(self.type_map_raw.keys())

    def get_feature_type_raw(self, feature: str) -> str:
        return self.type_map_raw[feature]

    def get_feature_types_special(self, feature: str) -> list:
        return self._get_feature_types(feature=feature, feature_types_dict=self.type_group_map_special)

    # TODO: Can remove, this is same output as self.type_map_raw
    def get_type_group_map_raw_flattened(self) -> dict:
        return {feature: type_family for type_family, features in self.type_group_map_raw.items() for feature in features}

    @staticmethod
    def get_type_group_map_raw_from_flattened(type_map_raw):
        type_group_map_raw = defaultdict(list)
        for feature, dtype in type_map_raw.items():
            type_group_map_raw[dtype].append(feature)
        return type_group_map_raw

    def remove_features(self, features: list, inplace=False):
        if inplace:
            metadata = self
        else:
            metadata = copy.deepcopy(self)
        features_invalid = [feature for feature in features if feature not in self.get_features()]
        if features_invalid:
            raise KeyError(f'remove_features was called with a feature that does not exist in feature metadata. Invalid Features: {features_invalid}')
        metadata._remove_features_from_type_map(d=metadata.type_map_raw, features=features)
        metadata._remove_features_from_type_group_map(d=metadata.type_group_map_raw, features=features)
        metadata._remove_features_from_type_group_map(d=metadata.type_group_map_special, features=features)
        return metadata

    def keep_features(self, features: list, inplace=False):
        '''Removes all features except for those in `features`'''
        features_invalid = [feature for feature in features if feature not in self.get_features()]
        if features_invalid:
            raise KeyError(f'keep_features was called with a feature that does not exist in feature metadata. Invalid Features: {features_invalid}')
        features_to_remove = [feature for feature in self.get_features() if feature not in features]
        return self.remove_features(features=features_to_remove, inplace=inplace)

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

        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    @staticmethod
    def _get_feature_types(feature: str, feature_types_dict: dict) -> list:
        feature_types = []
        for dtype_family in feature_types_dict:
            if feature in feature_types_dict[dtype_family]:
                feature_types.append(dtype_family)
        feature_types = sorted(feature_types)
        return feature_types

    # Joins a list of metadata objects together, returning a new metadata object
    @staticmethod
    def join_metadatas(metadata_list, allow_shared_raw_features=False):
        metadata_new = copy.deepcopy(metadata_list[0])
        for metadata in metadata_list[1:]:
            metadata_new = metadata_new.join_metadata(metadata, allow_shared_raw_features=allow_shared_raw_features)
        return metadata_new

    def _get_feature_metadata_full(self):
        feature_metadata_full = defaultdict(list)

        for feature in self.get_features():
            feature_type_raw = self.type_map_raw[feature]
            feature_types_special = tuple(self.get_feature_types_special(feature))
            feature_metadata_full[(feature_type_raw, feature_types_special)].append(feature)

        return feature_metadata_full

    def print_feature_metadata_full(self, log_prefix=''):
        feature_metadata_full = self._get_feature_metadata_full()
        if not feature_metadata_full:
            return
        keys = list(feature_metadata_full.keys())
        keys = sorted(keys)
        output = [((key[0], list(key[1])), feature_metadata_full[key]) for key in keys]
        max_key_len = max([len(str(key)) for key, _ in output])
        max_val_len = max([len(str(len(val))) for _, val in output])
        for key, val in output:
            key_len = len(str(key))
            val_len = len(str(len(val)))
            max_key_minus_cur = max(max_key_len - key_len, 0)
            max_val_minus_cur = max(max_val_len - val_len, 0)
            features = str(val[:3])
            if len(val) > 3:
                features = features[:-1] + ', ...]'
            if val:
                logger.log(20, f'{log_prefix}{key}{" " * max_key_minus_cur} : {" " * max_val_minus_cur}{len(val)} | {features}')
