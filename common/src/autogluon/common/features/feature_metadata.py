import copy
import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union

import pandas as pd

from .infer_types import get_type_group_map_special, get_type_map_raw

logger = logging.getLogger(__name__)


class FeatureMetadata:
    """
    Feature metadata contains information about features that are not directly apparent in the raw data itself.
    This enables feature generators to properly process features, and allows downstream models to properly handle features during training and inference.

    Parameters
    ----------
    type_map_raw : Dict[str, str]
        Dictionary of feature names to raw types.
        The values can be anything, but it is generally recommended they be one of:
            ['int', 'float', 'object', 'category', 'datetime']
    type_group_map_special : Dict[str, List[str]], optional
        Dictionary of special types to lists of feature names.
        The keys can be anything, but it is generally recommended they be one of:
            ['binned', 'datetime_as_int', 'datetime_as_object', 'text', 'text_as_category', 'text_special', 'text_ngram', 'image_path', 'stack']
        For descriptions of each special feature-type, see: `autogluon.common.features.types`
        Feature names that appear in the value lists must also be keys in type_map_raw.
        Feature names are not required to have special types.
        Only one of type_group_map_special and type_map_special can be specified.
    type_map_special : Dict[str, List[str]], optional
        Dictionary of feature names to lists of special types.
        This is an alternative representation of the special types.
        Only one of type_group_map_special and type_map_special can be specified.
    """

    def __init__(
        self,
        type_map_raw: Dict[str, str],
        type_group_map_special: Dict[str, List[str]] = None,
        type_map_special: Dict[str, List[str]] = None,
    ):
        if type_group_map_special is None:
            if type_map_special is not None:
                type_group_map_special = self.get_type_group_map_special_from_type_map_special(type_map_special)
            else:
                type_group_map_special = defaultdict(list)
        elif type_map_special is not None:
            raise ValueError("Only one of type_group_map_special and type_map_special can be specified in init.")
        if not isinstance(type_group_map_special, defaultdict):
            type_group_map_special = defaultdict(list, type_group_map_special)

        self.type_map_raw = type_map_raw
        self.type_group_map_special = type_group_map_special

        self._validate()

    def __eq__(self, other) -> bool:
        if set(self.type_map_raw.keys()) != set(other.type_map_raw.keys()):
            return False
        for k in self.type_map_raw.keys():
            if self.type_map_raw[k] != other.type_map_raw[k]:
                return False
        if set(self.type_group_map_special.keys()) != set(other.type_group_map_special.keys()):
            return False
        for k in self.type_group_map_special.keys():
            if set(self.type_group_map_special[k]) != set(other.type_group_map_special[k]):
                return False
        return True

    # Confirms if inputs are valid
    def _validate(self):
        type_group_map_special_expanded = []
        for key in self.type_group_map_special:
            type_group_map_special_expanded += self.type_group_map_special[key]

        features_invalid = []
        type_map_raw_keys = self.type_map_raw.keys()
        for feature in type_group_map_special_expanded:
            if feature not in type_map_raw_keys:
                features_invalid.append(feature)
        if features_invalid:
            raise AssertionError(
                f"{len(features_invalid)} features are present in type_group_map_special but not in type_map_raw. Invalid features: {features_invalid}"
            )

    # Note: This is not optimized for speed. Do not rely on this function during inference.
    # TODO: Add valid_names, invalid_names arguments which override all other arguments for the features listed?
    def get_features(
        self,
        valid_raw_types: list = None,
        valid_special_types: list = None,
        invalid_raw_types: list = None,
        invalid_special_types: list = None,
        required_special_types: list = None,
        required_raw_special_pairs: List[Tuple[str, Union[List[str], Set[str]]]] = None,
        required_exact=False,
        required_at_least_one_special=False,
    ) -> List[str]:
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
            Identical to getting the union of calling get_features(valid_raw_types=[raw_type], required_special_types=special_types) for every
            element of (raw_type, special_types) in required_raw_special_pairs
            If raw_type is None, then any feature will satisfy the raw type requirement.
            If special_types is None, then any feature will satisfy the special type requirement (including those with no special types).
        required_exact : bool, default False
            If True, then if a feature does not have the exact same special types (with no extra special types) as required_special_types, it is pruned.
            This is also applied to required_raw_special_pairs if specified.
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
            features = [
                feature
                for feature in features
                if not valid_special_types_set.isdisjoint(self.get_feature_types_special(feature))
                or not self.get_feature_types_special(feature)
            ]
        if invalid_raw_types is not None:
            features = [feature for feature in features if self.get_feature_type_raw(feature) not in invalid_raw_types]
        if invalid_special_types is not None:
            invalid_special_types_set = set(invalid_special_types)
            features = [
                feature
                for feature in features
                if invalid_special_types_set.isdisjoint(self.get_feature_types_special(feature))
            ]
        if required_special_types is not None:
            required_special_types_set = set(required_special_types)
            if required_exact:
                features = [
                    feature
                    for feature in features
                    if required_special_types_set == set(self.get_feature_types_special(feature))
                ]
            else:
                features = [
                    feature
                    for feature in features
                    if required_special_types_set.issubset(self.get_feature_types_special(feature))
                ]
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
            raise KeyError(f"{feature} does not exist in {self.__class__.__name__}.")
        return self._get_feature_types(feature=feature, feature_types_dict=self.type_group_map_special)

    def get_type_map_special(self) -> dict:
        return {feature: self.get_feature_types_special(feature) for feature in self.get_features()}

    @staticmethod
    def get_type_group_map_special_from_type_map_special(type_map_special: Dict[str, List[str]]):
        type_group_map_special = defaultdict(list)
        for feature in type_map_special:
            for type_special in type_map_special[feature]:
                type_group_map_special[type_special].append(feature)
        return type_group_map_special

    def get_type_group_map_raw(self):
        type_group_map_raw = defaultdict(list)
        for feature, dtype in self.type_map_raw.items():
            type_group_map_raw[dtype].append(feature)
        return type_group_map_raw

    def remove_features(self, features: list, inplace=False):
        """Removes all features from metadata that are in features"""
        if inplace:
            metadata = self
        else:
            metadata = copy.deepcopy(self)
        features_invalid = [feature for feature in features if feature not in self.get_features()]
        if features_invalid:
            raise KeyError(
                f"remove_features was called with a feature that does not exist in feature metadata. Invalid Features: {features_invalid}"
            )
        metadata._remove_features_from_type_map(d=metadata.type_map_raw, features=features)
        metadata._remove_features_from_type_group_map(d=metadata.type_group_map_special, features=features)
        return metadata

    def keep_features(self, features: list, inplace=False):
        """Removes all features from metadata except for those in features"""
        features_invalid = [feature for feature in features if feature not in self.get_features()]
        if features_invalid:
            raise KeyError(
                f"keep_features was called with a feature that does not exist in feature metadata. Invalid Features: {features_invalid}"
            )
        features_to_remove = [feature for feature in self.get_features() if feature not in features]
        return self.remove_features(features=features_to_remove, inplace=inplace)

    def add_special_types(self, type_map_special: Dict[str, List[str]], inplace=False):
        """
        Adds special types to features.

        Parameters
        ----------
        type_map_special : Dict[str, List[str]]
            Dictionary of feature -> list of special types to add.
            Features in dictionary must already exist in the FeatureMetadata object.
        inplace : bool, default False
            If True, updates self inplace and returns self.
            If False, updates a copy of self and returns copy.
        Returns
        -------
        :class:`FeatureMetadata` object.

        Examples
        --------
        >>> from autogluon.common.features.feature_metadata import FeatureMetadata
        >>> feature_metadata = FeatureMetadata({'FeatureA': 'int', 'FeatureB': 'object'})
        >>> feature_metadata = feature_metadata.add_special_types({'FeatureA': ['MySpecialType'], 'FeatureB': ['MySpecialType', 'text']})
        """
        if inplace:
            metadata = self
        else:
            metadata = copy.deepcopy(self)
        valid_features = set(self.get_features())

        for feature, special_types in type_map_special.items():
            if feature not in valid_features:
                raise ValueError(
                    f'"{feature}" does not exist in this FeatureMetadata object. Only existing features can be assigned special types.'
                )
            for special_type in special_types:
                metadata.type_group_map_special[special_type].append(feature)
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

    def rename_features(self, rename_map: dict, inplace=False):
        """Rename all features from metadata that are keys in rename_map to their values."""
        if inplace:
            metadata = self
        else:
            metadata = copy.deepcopy(self)
        before_len = len(metadata.type_map_raw.keys())
        metadata.type_map_raw = {rename_map.get(key, key): val for key, val in metadata.type_map_raw.items()}
        after_len = len(metadata.type_map_raw.keys())
        if before_len != after_len:
            raise AssertionError(
                "key names conflicted during renaming. Do not rename features to exist feature names."
            )
        for dtype in metadata.type_group_map_special:
            metadata.type_group_map_special[dtype] = [
                rename_map.get(feature, feature) for feature in metadata.type_group_map_special[dtype]
            ]
        return metadata

    # TODO: Add documentation on shared_raw_features usage
    def join_metadata(self, metadata, shared_raw_features="error"):
        """Join two FeatureMetadata objects together, returning a new FeatureMetadata object"""
        if shared_raw_features not in ["error", "error_if_diff", "overwrite"]:
            raise ValueError(
                f"shared_raw_features must be one of {['error', 'error_if_diff', 'overwrite']}, but was: '{shared_raw_features}'"
            )
        type_map_raw = copy.deepcopy(self.type_map_raw)
        shared_features = []
        shared_features_diff_types = []
        for key, features in metadata.type_map_raw.items():
            if key in type_map_raw:
                shared_features.append(key)
                if type_map_raw[key] != metadata.type_map_raw[key]:
                    shared_features_diff_types.append(key)
        if shared_features:
            if shared_raw_features == "error":
                logger.error("ERROR: Conflicting metadata:")
                logger.error("Metadata 1:")
                self.print_feature_metadata_full(log_prefix="\t", log_level=40)
                logger.error("Metadata 2:")
                metadata.print_feature_metadata_full(log_prefix="\t", log_level=40)
                raise AssertionError(
                    f"Metadata objects to join share raw features, but `shared_raw_features='error'`. Shared features: {shared_features}"
                )
            if shared_features_diff_types:
                if shared_raw_features == "overwrite":
                    logger.log(
                        20,
                        f"Overwriting type_map_raw during FeatureMetadata join. "
                        f"Shared features with conflicting types: {shared_features_diff_types}",
                    )
                    shared_features = []
                elif shared_raw_features == "error_if_diff":
                    logger.error("ERROR: Conflicting metadata:")
                    logger.error("Metadata 1:")
                    self.print_feature_metadata_full(log_prefix="\t", log_level=40)
                    logger.error("Metadata 2:")
                    metadata.print_feature_metadata_full(log_prefix="\t", log_level=40)
                    raise AssertionError(
                        f"Metadata objects to join share raw features but do not agree on raw dtypes, "
                        f"and `shared_raw_features='error_if_diff'`. Shared conflicting features: {shared_features_diff_types}"
                    )
        type_map_raw.update({key: val for key, val in metadata.type_map_raw.items() if key not in shared_features})

        type_group_map_special = self._add_type_group_map_special(
            [self.type_group_map_special, metadata.type_group_map_special]
        )

        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    @staticmethod
    def _add_type_group_map_special(type_group_map_special_lst: List[dict]) -> dict:
        if not type_group_map_special_lst:
            return defaultdict(list)
        type_group_map_special_combined = copy.deepcopy(type_group_map_special_lst[0])
        for type_group_map_special in type_group_map_special_lst[1:]:
            for key, features in type_group_map_special.items():
                if key in type_group_map_special_combined:
                    features_to_add = [
                        feature for feature in features if feature not in type_group_map_special_combined[key]
                    ]
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
    def join_metadatas(metadata_list, shared_raw_features="error"):
        metadata_new = copy.deepcopy(metadata_list[0])
        for metadata in metadata_list[1:]:
            metadata_new = metadata_new.join_metadata(metadata, shared_raw_features=shared_raw_features)
        return metadata_new

    def to_dict(self, inverse=False) -> dict:
        if not inverse:
            feature_metadata_dict: Dict[Union[str, Tuple[str, tuple]], Any] = dict()
        else:
            feature_metadata_dict = defaultdict(list)

        for feature in self.get_features():
            feature_type_raw = self.type_map_raw[feature]
            feature_types_special = tuple(self.get_feature_types_special(feature))
            if not inverse:
                feature_metadata_dict[feature] = (feature_type_raw, feature_types_special)
            else:
                feature_metadata_dict[(feature_type_raw, feature_types_special)].append(feature)

        if inverse:
            feature_metadata_dict = dict(feature_metadata_dict)

        return feature_metadata_dict

    def print_feature_metadata_full(
        self, log_prefix="", print_only_one_special=False, log_level=20, max_list_len=5, return_str=False
    ):
        feature_metadata_dict = self.to_dict(inverse=True)
        if not feature_metadata_dict:
            if return_str:
                return ""
            else:
                return
        keys = list(feature_metadata_dict.keys())
        keys = sorted(keys)
        output = [((key[0], list(key[1])), feature_metadata_dict[key]) for key in keys]
        output_str = ""
        if print_only_one_special:
            for i, ((raw, special), features) in enumerate(output):
                if len(special) == 1:
                    output[i] = ((raw, special[0]), features)
                elif len(special) > 1:
                    output[i] = ((raw, special[0]), features)
                    logger.warning(
                        f"Warning: print_only_one_special=True was set, but features with {len(special)} special types were found. "
                        f"Invalid Types: {output[i]}"
                    )
                else:
                    output[i] = ((raw, None), features)
        max_key_len = max([len(str(key)) for key, _ in output])
        max_val_len = max([len(str(len(val))) for _, val in output])
        for key, val in output:
            key_len = len(str(key))
            val_len = len(str(len(val)))
            max_key_minus_cur = max(max_key_len - key_len, 0)
            max_val_minus_cur = max(max_val_len - val_len, 0)
            if max_list_len is not None:
                features = str(val[:max_list_len])
                if len(val) > max_list_len:
                    features = features[:-1] + ", ...]"
            else:
                features = str(val)
            if val:
                message = (
                    f'{log_prefix}{key}{" " * max_key_minus_cur} : {" " * max_val_minus_cur}{len(val)} | {features}'
                )
                if return_str:
                    output_str += message + "\n"
                else:
                    logger.log(log_level, message)
        if return_str:
            if output_str[-1] == "\n":
                output_str = output_str[:-1]
            return output_str

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        """
        Construct FeatureMetadata based on the inferred feature types of an input :class:`pd.DataFrame`.

        Parameters
        ----------
        df : :class:`pd.DataFrame`
            DataFrame used to infer FeatureMetadata.

        Returns
        -------
        :class:`FeatureMetadata` object.
        """
        type_map_raw = get_type_map_raw(df)
        type_group_map_special = get_type_group_map_special(df)
        return cls(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    def verify_data(self, df: pd.DataFrame) -> bool:
        """
        Returns True if the DataFrame's raw types match FeatureMetadata, else False
        """
        type_map_raw = get_type_map_raw(df)
        features = set(type_map_raw.keys())
        if features != set(self.get_features()):
            return False
        return self._verify_data_type_raw(type_map_raw=type_map_raw)

    def verify_data_subset(self, df: pd.DataFrame) -> bool:
        """
        Returns True if the DataFrame's features are a subset of FeatureMetadata and its raw types match FeatureMetadata, else False
        """
        type_map_raw = get_type_map_raw(df)
        features = set(type_map_raw.keys())
        if not features.issubset(set(self.get_features())):
            return False
        return self._verify_data_type_raw(type_map_raw=type_map_raw)

    def verify_data_superset(self, df: pd.DataFrame) -> bool:
        """
        Returns True if the DataFrame's features are a superset of FeatureMetadata and its raw types match FeatureMetadata, else False
        """
        type_map_raw = get_type_map_raw(df)
        features = set(type_map_raw.keys())
        features_self = set(self.get_features())
        if not features.issuperset(features_self):
            return False
        type_map_raw = {k: v for k, v in type_map_raw.items() if k in features_self}
        return self._verify_data_type_raw(type_map_raw=type_map_raw)

    def _verify_data_type_raw(self, type_map_raw: dict) -> bool:
        features = set(type_map_raw.keys())
        for feature in features:
            if type_map_raw[feature] != self.type_map_raw[feature]:
                return False
        return True

    def __str__(self):
        return self.print_feature_metadata_full(return_str=True)
