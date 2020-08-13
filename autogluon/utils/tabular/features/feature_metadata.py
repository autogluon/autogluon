import copy
import logging
from collections import defaultdict

from .types import get_type_map_raw, get_type_group_map_special

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

    def rename_features(self, rename_map: dict, inplace=False):
        if inplace:
            metadata = self
        else:
            metadata = copy.deepcopy(self)
        metadata.type_map_raw = {rename_map.get(key, key): val for key, val in metadata.type_map_raw.items()}
        metadata.type_group_map_raw = metadata.get_type_group_map_raw_from_flattened(type_map_raw=metadata.type_map_raw)
        for type in metadata.type_group_map_special:
            metadata.type_group_map_special[type] = [rename_map.get(feature, feature) for feature in metadata.type_group_map_special[type]]
        return metadata

    # Joins two metadata objects together, returning a new metadata object
    # TODO: Add documentation on shared_raw_features usage
    def join_metadata(self, metadata, shared_raw_features='error'):
        if shared_raw_features not in ['error', 'error_if_diff', 'overwrite']:
            raise ValueError(f"shared_raw_features must be one of {['error', 'error_if_diff', 'overwrite']}, but was: '{shared_raw_features}'")
        type_map_raw = copy.deepcopy(self.type_map_raw)
        shared_features = []
        shared_features_diff_types = []
        for key, features in metadata.type_map_raw.items():
            if key in type_map_raw:
                shared_features.append(key)
                if type_map_raw[key] != metadata.type_map_raw[key]:
                    shared_features_diff_types.append(key)
        if shared_features:
            if shared_raw_features == 'error':
                logger.error('ERROR: Conflicting metadata:')
                logger.error('Metadata 1:')
                self.print_feature_metadata_full(log_prefix='\t', log_level=40)
                logger.error('Metadata 2:')
                metadata.print_feature_metadata_full(log_prefix='\t', log_level=40)
                raise AssertionError(f"Metadata objects to join share raw features, but `shared_raw_features='error'`. Shared features: {shared_features}")
            if shared_features_diff_types:
                if shared_raw_features == 'overwrite':
                    logger.log(20, f'Overwriting type_map_raw during FeatureMetadata join. Shared features with conflicting types: {shared_features_diff_types}')
                    shared_features = []
                elif shared_raw_features == 'error_if_diff':
                    logger.error('ERROR: Conflicting metadata:')
                    logger.error('Metadata 1:')
                    self.print_feature_metadata_full(log_prefix='\t', log_level=40)
                    logger.error('Metadata 2:')
                    metadata.print_feature_metadata_full(log_prefix='\t', log_level=40)
                    raise AssertionError(f"Metadata objects to join share raw features but do not agree on raw dtypes, and `shared_raw_features='error_if_diff'`. Shared conflicting features: {shared_features_diff_types}")
        type_map_raw.update({key: val for key, val in metadata.type_map_raw.items() if key not in shared_features})

        type_group_map_special = self.add_type_group_map_special([self.type_group_map_special, metadata.type_group_map_special])

        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    @staticmethod
    def add_type_group_map_special(type_group_map_special_lst: list):
        if not type_group_map_special_lst:
            return defaultdict(list)
        type_group_map_special_combined = copy.deepcopy(type_group_map_special_lst[0])
        for type_group_map_special in type_group_map_special_lst[1:]:
            for key, features in type_group_map_special.items():
                if key in type_group_map_special_combined:
                    features_to_add = [feature for feature in features if feature not in type_group_map_special_combined[key]]
                    type_group_map_special_combined[key] += features_to_add
                else:
                    type_group_map_special_combined[key] = features
        return type_group_map_special_combined

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
    def join_metadatas(metadata_list, shared_raw_features='error'):
        metadata_new = copy.deepcopy(metadata_list[0])
        for metadata in metadata_list[1:]:
            metadata_new = metadata_new.join_metadata(metadata, shared_raw_features=shared_raw_features)
        return metadata_new

    def _get_feature_metadata_full(self):
        feature_metadata_full = defaultdict(list)

        for feature in self.get_features():
            feature_type_raw = self.type_map_raw[feature]
            feature_types_special = tuple(self.get_feature_types_special(feature))
            feature_metadata_full[(feature_type_raw, feature_types_special)].append(feature)

        return feature_metadata_full

    def print_feature_metadata_full(self, log_prefix='', print_only_one_special=False, log_level=20):
        feature_metadata_full = self._get_feature_metadata_full()
        if not feature_metadata_full:
            return
        keys = list(feature_metadata_full.keys())
        keys = sorted(keys)
        output = [((key[0], list(key[1])), feature_metadata_full[key]) for key in keys]
        if print_only_one_special:
            for i, ((raw, special), features) in enumerate(output):
                if len(special) == 1:
                    output[i] = ((raw, special[0]), features)
                elif len(special) > 1:
                    output[i] = ((raw, special[0]), features)
                    logger.warning(f'Warning: print_only_one_special=True was set, but features with {len(special)} special types were found. Invalid Types: {output[i]}')
                else:
                    output[i] = ((raw, None), features)
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
                logger.log(log_level, f'{log_prefix}{key}{" " * max_key_minus_cur} : {" " * max_val_minus_cur}{len(val)} | {features}')

    @classmethod
    def from_df(cls, df):
        type_map_raw = get_type_map_raw(df)
        type_group_map_special = get_type_group_map_special(df)
        return cls(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)
