import copy
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

from .types import get_type_map_raw, get_type_group_map_special

logger = logging.getLogger(__name__)

# Raw types: Raw data type information grouped into families.
# For example: uint8, int8, int16, int32, and int64 features all map to 'int'
R_INT = 'int'
R_FLOAT = 'float'
R_OBJECT = 'object'
R_CATEGORY = 'category'
R_DATETIME = 'datetime'
# TODO: R_BOOL/R_BOOLEAN?
# TODO: R_FLOAT_SPARSE/R_INT_SPARSE/R_CATEGORY_SPARSE?

# Special types: Meta information about the special meaning of a feature that is not present in the raw data.
# feature has been binned into discrete integer values from its original representation
S_BINNED = 'binned'

# feature was originally a datetime type that was converted to numeric
S_DATETIME_AS_INT = 'datetime_as_int'

# feature is a datetime in object form (string dates), which can be converted to datetime via pd.to_datetime
S_DATETIME_AS_OBJECT = 'datetime_as_object'

# feature is an object type that contains text information that can be utilized in natural language processing
S_TEXT = 'text'

# feature is a categorical that was originally text information. It may or may not still contain the raw text in its data.
S_TEXT_AS_CATEGORY = 'text_as_category'

# feature is a generated feature based off of a text feature but is not an ngram. Examples include character count, word count, symbol count, etc.
S_TEXT_SPECIAL = 'text_special'

# feature is a generated feature based off of a text feature that is an ngram.
S_TEXT_NGRAM = 'text_ngram'

# feature is a generated feature based off of a ML model's prediction probabilities of the label column for the row.
# Any model which takes a stack feature as input is a stack ensemble.
S_STACK = 'stack'


class FeatureMetadata:
    """
    Contains feature type metadata information such as type family groups (type_group_map_raw) and special feature type groups (type_group_map_special)

    # TODO: rewrite
    type_group_map_raw is the dictionary computed as output to :function:`autogluon.utils.tabular.features.utils.get_type_group_map_raw`
    type_group_map_special is an optional dictionary to communicate special properties of features to downstream models that have special handling functionality for those feature types.
        As an example, type_group_map_special might contain a key 'text_ngram' indicating that the list of values are all features which were generated from a nlp vectorizer and represent ngrams.
        A downstream model such as a K-Nearest-Neighbor model could then check if 'text_ngram' is present in type_group_map_special and drop those features if present, to speed up training and inference time.
    """
    def __init__(self, type_map_raw: Dict[str, str], type_group_map_special: Dict[str, List[str]] = None):
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

    # Note: This is not optimized for speed. Do not rely on this function during inference.
    def get_features(self, valid_raw_types: list = None, valid_special_types: list = None, invalid_raw_types: list = None, invalid_special_types: list = None,
                     required_special_types: list = None, required_raw_special_pairs: List[Tuple[str, List[str]]] = None, required_exact=False, required_at_least_one_special=False):
        """
        Returns a list of features held within the feature metadata object after being pruned through the available parameters.

         Parameters
        ----------
        valid_raw_types : list, default None
            If a feature's raw type is not in this list, it is pruned.
            If None, then no features are pruned through this logic.
        valid_special_types : list, default None
            If a feature has a special type not in this list, it is pruned.
            Features without special types are never pruned through this logic.
            If None, then no features are pruned through this logic.
        invalid_raw_types : list, default None
            If a feature's raw type is in this list, it is pruned.
            If None, then no features are pruned through this logic.
        invalid_special_types : list, default None
            If a feature has a special type in this list, it is pruned.
            Features without special types are never pruned through this logic.
            If None, then no features are pruned through this logic.
        required_special_types : list, default None
            If a feature does not have all of the special types in this list, it is pruned.
            Features without special types are pruned through this logic.
            If None, then no features are pruned through this logic.
        required_raw_special_pairs : List[Tuple[str, List[str]]], default None
            If a feature does not satisfy the (raw_type, special_types) requirement of at least one of the elements in this list, it is pruned.
            Identical to getting the union of calling get_features(valid_raw_types=[raw_type], required_special_types=special_types) for every element of (raw_type, special_types) in required_raw_special_pairs
            If raw_type is None, then any feature will satisfy the raw type requirement.
            If special_types is None, then any feature will satisfy the special type requirement (including those with no special types).
        required_exact : bool, default False
            If True, then if a feature does not have the exact same special types (with no extra special types) as required_special_types, it is pruned.
            This also applied to required_raw_special_pairs if specified.
            Has no effect if required_special_types and required_raw_special_pairs are None.
        required_at_least_one_special : bool, default False
            If True, then if a feature has zero special types, it is pruned.

        Returns
        -------
        features : list of feature names in feature metadata that satisfy all checks dictated by the parameters.

        """
        features = list(self.type_map_raw.keys())

        if valid_raw_types is not None:
            features = [feature for feature in features if self.get_feature_type_raw(feature) in valid_raw_types]
        if valid_special_types is not None:
            valid_special_types_set = set(valid_special_types)
            features = [feature for feature in features if not valid_special_types_set.isdisjoint(self.get_feature_types_special(feature)) or not self.get_feature_types_special(feature)]
        if invalid_raw_types is not None:
            features = [feature for feature in features if self.get_feature_type_raw(feature) not in invalid_raw_types]
        if invalid_special_types is not None:
            invalid_special_types_set = set(invalid_special_types)
            features = [feature for feature in features if invalid_special_types_set.isdisjoint(self.get_feature_types_special(feature))]
        if required_special_types is not None:
            required_special_types_set = set(required_special_types)
            if required_exact:
                features = [feature for feature in features if required_special_types_set == set(self.get_feature_types_special(feature))]
            else:
                features = [feature for feature in features if required_special_types_set.issubset(self.get_feature_types_special(feature))]
        if required_at_least_one_special:
            features = [feature for feature in features if self.get_feature_types_special(feature)]
        if required_raw_special_pairs is not None:
            features_og = copy.deepcopy(features)
            features_to_keep = []
            for valid_raw, valid_special in required_raw_special_pairs:
                if valid_special is not None:
                    valid_special = set(valid_special)
                features_to_keep_inner = []
                for feature in features:
                    feature_type_raw = self.get_feature_type_raw(feature)
                    feature_types_special = set(self.get_feature_types_special(feature))
                    if valid_raw is None or feature_type_raw == valid_raw:
                        if valid_special is None:
                            features_to_keep_inner.append(feature)
                        elif required_exact:
                            if valid_special == feature_types_special:
                                features_to_keep_inner.append(feature)
                        elif valid_special.issubset(feature_types_special):
                            features_to_keep_inner.append(feature)
                features = [feature for feature in features if feature not in features_to_keep_inner]
                features_to_keep += features_to_keep_inner
            features = [feature for feature in features_og if feature in features_to_keep]

        return features

    def get_feature_type_raw(self, feature: str) -> str:
        return self.type_map_raw[feature]

    def get_feature_types_special(self, feature: str) -> list:
        if feature not in self.type_map_raw:
            raise KeyError(f'{feature} does not exist in {self.__class__.__name__}.')
        return self._get_feature_types(feature=feature, feature_types_dict=self.type_group_map_special)

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
        """Removes all features except for those in `features`"""
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
        before_len = len(metadata.type_map_raw.keys())
        metadata.type_map_raw = {rename_map.get(key, key): val for key, val in metadata.type_map_raw.items()}
        after_len = len(metadata.type_map_raw.keys())
        if before_len != after_len:
            raise AssertionError(f'key names conflicted during renaming. Do not rename features to exist feature names.')
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

        type_group_map_special = self._add_type_group_map_special([self.type_group_map_special, metadata.type_group_map_special])

        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    @staticmethod
    def _add_type_group_map_special(type_group_map_special_lst: List[dict]) -> dict:
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

    def get_feature_metadata_full(self) -> dict:
        feature_metadata_full = defaultdict(list)

        for feature in self.get_features():
            feature_type_raw = self.type_map_raw[feature]
            feature_types_special = tuple(self.get_feature_types_special(feature))
            feature_metadata_full[(feature_type_raw, feature_types_special)].append(feature)

        feature_metadata_full = dict(feature_metadata_full)

        return feature_metadata_full

    def print_feature_metadata_full(self, log_prefix='', print_only_one_special=False, log_level=20):
        feature_metadata_full = self.get_feature_metadata_full()
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
