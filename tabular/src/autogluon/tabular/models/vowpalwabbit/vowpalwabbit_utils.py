import pandas as pd

from autogluon.common.features.types import R_CATEGORY, R_FLOAT, R_INT, S_TEXT


class VWFeaturesConverter:
    """
    Converts features in PandasDataFrame to VW format
    Ref: https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format
    """

    PIPE = "|"
    SPACE = " "

    # TODO: Add support for different namespaces

    def convert_features_to_vw_format(self, X, feature_metadata) -> pd.Series:
        """
        Converts features to VW format.
        :param X: features
        :param feature_metadata: schema of X
        :return: Returns a series of features converted to VW format
        """

        X_out: pd.Series = None

        for feature in feature_metadata:
            raw_feature, special_feature = feature_metadata[feature]
            if X_out is None:
                X_out = (
                    self.PIPE
                    + self.SPACE
                    + self.__generate_namespace_based_on_ml_type(X[feature], raw_feature, special_feature, feature).astype("str")
                    + self.SPACE
                )
            else:
                X_out += (
                    "" + self.SPACE + self.__generate_namespace_based_on_ml_type(X[feature], raw_feature, special_feature, feature).astype("str") + self.SPACE
                )
        return X_out

    def __generate_namespace_based_on_ml_type(self, input_series, raw_feature, special_feature, feature_name=None):
        """
        Based on the type of feature, preprocess/sanify these features so that it is in VW format
        Only use raw text, numeric integer, numeric decimals, and category
        Ref: https://github.com/autogluon/autogluon/blob/master/common/src/autogluon/common/features/types.py

        :param input_series: A single feature as Pandas Series
        :param raw_feature: Raw feature Type
        :param special_feature: Special Feature Type
        :param feature_name: Column Name of this feature
        :return: Preprocessed Feature as a Pandas Series
        """
        if S_TEXT in special_feature:
            return input_series.apply(self.__preprocess_text)
        elif raw_feature in [R_INT, R_FLOAT]:
            return input_series.apply(self.__numeric_namespace_generator, args=(feature_name,))
        elif raw_feature == R_CATEGORY:
            return input_series.apply(self.__categorical_namespace_generator, args=(feature_name,))
        else:
            raise ValueError(
                f"Received unsupported raw_feature_type '{str(raw_feature)}' special_feature_type '{str(special_feature)}'"
                f" for feature '{feature_name}' for class {self.__class__.__name__}."
            )

    def __preprocess_text(self, s) -> str:
        if pd.isnull(s):
            return ""
        s = " ".join(str(s).split())
        # Added split to remove tabs spaces since tab is used as separator
        text = self.__sanify(s)
        return text

    def __sanify(self, s) -> str:
        """
        The sanify is performed because : and | are reserved by vowpal wabbit for distinguishing namespaces and numeric
        data
        @param s: input string
        @returns string
        """
        return str(s).replace(":", ";").replace("|", "/")

    def __numeric_namespace_generator(self, feature, feature_name) -> str:
        if pd.isnull(feature):
            return ""
        return feature_name + ":" + str(feature)

    def __categorical_namespace_generator(self, feature, feature_name) -> str:
        if pd.isnull(feature):
            return ""
        else:
            feature = str(feature).replace(" ", "_")
            return feature_name + "=" + self.__sanify(feature)
