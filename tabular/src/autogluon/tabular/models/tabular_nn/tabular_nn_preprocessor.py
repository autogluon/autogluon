from collections import OrderedDict

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer, FunctionTransformer  # PowerTransformer

from .categorical_encoders import OneHotMergeRaresHandleUnknownEncoder, OrdinalMergeRaresHandleUnknownEncoder


class TabularNeuralNetPreprocessor(object):

    def __init__(self, types_of_features, unique_category_str, impute_strategy, max_category_levels):
        self._types_of_features = types_of_features
        self.unique_category_str = unique_category_str
        self.impute_strategy = impute_strategy
        self.max_category_levels = max_category_levels
        self.processor = self._create_preprocessor()

    def _create_preprocessor(self):
        """ Defines data encoders used to preprocess different data types and creates instance variable which is sklearn ColumnTransformer object """
        continuous_features = self._types_of_features['continuous']
        skewed_features = self._types_of_features['skewed']
        onehot_features = self._types_of_features['onehot']
        embed_features = self._types_of_features['embed']
        language_features = self._types_of_features['language']
        transformers = []  # order of various column transformers in this list is important!
        if continuous_features:
            continuous_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.impute_strategy)),
                ('scaler', StandardScaler())])
            transformers.append(('continuous', continuous_transformer, continuous_features))
        if skewed_features:
            power_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.impute_strategy)),
                ('quantile', QuantileTransformer(output_distribution='normal'))])  # Or output_distribution = 'uniform'
            transformers.append(('skewed', power_transformer, skewed_features))
        if onehot_features:
            onehot_transformer = Pipeline(steps=[
                # TODO: Consider avoiding converting to string for improved memory efficiency
                ('to_str', FunctionTransformer(convert_df_dtype_to_str)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=self.unique_category_str)),
                ('onehot', OneHotMergeRaresHandleUnknownEncoder(max_levels=self.max_category_levels,
                                                                sparse=False))])  # test-time unknown values will be encoded as all zeros vector
            transformers.append(('onehot', onehot_transformer, onehot_features))
        if embed_features:  # Ordinal transformer applied to convert to-be-embedded categorical features to integer levels
            ordinal_transformer = Pipeline(steps=[
                ('to_str', FunctionTransformer(convert_df_dtype_to_str)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=self.unique_category_str)),
                ('ordinal', OrdinalMergeRaresHandleUnknownEncoder(
                    max_levels=self.max_category_levels))])  # returns 0-n when max_category_levels = n-1. category n is reserved for unknown test-time categories.
            transformers.append(('ordinal', ordinal_transformer, embed_features))
        if language_features:
            raise NotImplementedError("language_features cannot be used at the moment")
        return ColumnTransformer(
            transformers=transformers)  # numeric features are processed in the same order as in numeric_features vector, so feature-names remain the same.

    def get_feature_arraycol_map(self, max_category_levels):
        """ Returns OrderedDict of feature-name -> list of column-indices in processed data array corresponding to this feature """
        feature_preserving_transforms = set(['continuous', 'skewed', 'ordinal', 'language'])  # these transforms do not alter dimensionality of feature
        feature_arraycol_map = {}  # unordered version
        current_colindex = 0
        for transformer in self.processor.transformers_:
            transformer_name = transformer[0]
            transformed_features = transformer[2]
            if transformer_name in feature_preserving_transforms:
                for feature in transformed_features:
                    if feature in feature_arraycol_map:
                        raise ValueError("same feature is processed by two different column transformers: %s" % feature)
                    feature_arraycol_map[feature] = [current_colindex]
                    current_colindex += 1
            elif transformer_name == 'onehot':
                oh_encoder = [step for (name, step) in transformer[1].steps if name == 'onehot'][0]
                for i in range(len(transformed_features)):
                    feature = transformed_features[i]
                    if feature in feature_arraycol_map:
                        raise ValueError("same feature is processed by two different column transformers: %s" % feature)
                    oh_dimensionality = min(len(oh_encoder.categories_[i]), max_category_levels+1)
                    feature_arraycol_map[feature] = list(range(current_colindex, current_colindex+oh_dimensionality))
                    current_colindex += oh_dimensionality
            else:
                raise ValueError("unknown transformer encountered: %s" % transformer_name)
        return OrderedDict([(key, feature_arraycol_map[key]) for key in feature_arraycol_map])

    # ------- Delegation methods --------

    def fit_transform(self, df):
        return self.processor.fit_transform(df)

    def transform(self, df):
        return self.processor.transform(df)


def convert_df_dtype_to_str(df):
    return df.astype(str)
