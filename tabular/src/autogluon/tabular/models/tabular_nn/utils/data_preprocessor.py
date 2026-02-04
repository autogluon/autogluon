"""Data preprocessing helper functions for tabular neural network models"""

from collections import OrderedDict

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (  # can also consider: PowerTransformer
    FunctionTransformer,
    QuantileTransformer,
    StandardScaler,
)

from .categorical_encoders import OneHotMergeRaresHandleUnknownEncoder, OrdinalMergeRaresHandleUnknownEncoder


def create_preprocessor(
    impute_strategy,
    max_category_levels,
    unique_category_str,
    continuous_features,
    skewed_features,
    onehot_features,
    embed_features,
    bool_features,
):
    """Creates sklearn ColumnTransformer that can be fit to training data to preprocess it for tabular neural network."""
    transformers = []  # order of various column transformers in this list is important!
    if continuous_features:
        continuous_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy=impute_strategy)), ("scaler", StandardScaler())]
        )
        transformers.append(("continuous", continuous_transformer, continuous_features))
    if skewed_features:
        power_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=impute_strategy)),
                ("quantile", QuantileTransformer(output_distribution="normal")),
            ]
        )  # Or output_distribution = 'uniform'
        transformers.append(("skewed", power_transformer, skewed_features))
    if onehot_features:
        onehot_transformer = Pipeline(
            steps=[("onehot", OneHotMergeRaresHandleUnknownEncoder(max_levels=max_category_levels, sparse=False))]
        )  # test-time unknown values will be encoded as all zeros vector
        transformers.append(("onehot", onehot_transformer, onehot_features))
    if embed_features:  # Ordinal transformer applied to convert to-be-embedded categorical features to integer levels
        ordinal_transformer = Pipeline(
            steps=[("ordinal", OrdinalMergeRaresHandleUnknownEncoder(max_levels=max_category_levels))]
        )  # returns 0-n when max_category_levels = n-1. category n is reserved for unknown test-time categories.
        transformers.append(("ordinal", ordinal_transformer, embed_features))
    try:
        out = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            force_int_remainder_cols=False,
        )  # numeric features are processed in the same order as in numeric_features vector, so feature-names remain the same.
    except:
        # TODO: Avoid try/except once scikit-learn 1.5 is minimum
        # Needed for scikit-learn 1.4 and 1.9+, force_int_remainder_cols is deprecated in 1.7 and introduced in 1.5
        # ref: https://github.com/autogluon/autogluon/issues/5289
        out = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
        )  # numeric features are processed in the same order as in numeric_features vector, so feature-names remain the same.
    return out


def convert_df_dtype_to_str(df):
    return df.astype(str)


def get_feature_arraycol_map(processor, max_category_levels):
    """Returns OrderedDict of feature-name -> list of column-indices in processed data array corresponding to this feature"""
    feature_preserving_transforms = set(
        ["continuous", "skewed", "ordinal", "bool", "remainder"]
    )  # these transforms do not alter dimensionality of feature
    feature_arraycol_map = {}  # unordered version
    current_colindex = 0
    for transformer in processor.transformers_:
        transformer_name = transformer[0]
        transformed_features = transformer[2]
        if transformer_name in feature_preserving_transforms:
            for feature in transformed_features:
                if feature in feature_arraycol_map:
                    raise ValueError("same feature is processed by two different column transformers: %s" % feature)
                feature_arraycol_map[feature] = [current_colindex]
                current_colindex += 1
        elif transformer_name == "onehot":
            oh_encoder = [step for (name, step) in transformer[1].steps if name == "onehot"][0]
            for i in range(len(transformed_features)):
                feature = transformed_features[i]
                if feature in feature_arraycol_map:
                    raise ValueError("same feature is processed by two different column transformers: %s" % feature)
                oh_dimensionality = min(len(oh_encoder.categories_[i]), max_category_levels + 1)
                feature_arraycol_map[feature] = list(range(current_colindex, current_colindex + oh_dimensionality))
                current_colindex += oh_dimensionality
        else:
            raise ValueError("unknown transformer encountered: %s" % transformer_name)
    return OrderedDict([(key, feature_arraycol_map[key]) for key in feature_arraycol_map])


def get_feature_type_map(feature_arraycol_map, types_of_features):
    """Returns OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed')."""
    if feature_arraycol_map is None:
        raise ValueError(
            "Must first call get_feature_arraycol_map() to set feature_arraycol_map before calling get_feature_type_map()"
        )
    vector_features = (
        types_of_features["continuous"]
        + types_of_features["skewed"]
        + types_of_features["onehot"]
        + types_of_features["bool"]
    )
    feature_type_map = OrderedDict()
    for feature_name in feature_arraycol_map:
        if feature_name in vector_features:
            feature_type_map[feature_name] = "vector"
        elif feature_name in types_of_features["embed"]:
            feature_type_map[feature_name] = "embed"
        else:
            feature_type_map[feature_name] = "vector"
    return feature_type_map
