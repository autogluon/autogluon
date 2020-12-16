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

    # ------- Delegation methods --------

    def fit_transform(self, df):
        return self.processor.fit_transform(df)

    def transform(self, df):
        return self.processor.transform(df)

    @property
    def transformers(self):
        return self.processor.transformers_


def convert_df_dtype_to_str(df):
    return df.astype(str)
