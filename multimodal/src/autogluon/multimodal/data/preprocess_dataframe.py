import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from nptyping import NDArray
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from autogluon.features import CategoryFeatureGenerator

from ..constants import (
    AUTOMM,
    CATEGORICAL,
    IDENTIFIER,
    IMAGE,
    IMAGE_PATH,
    LABEL,
    NER,
    NER_ANNOTATION,
    NULL,
    NUMERICAL,
    ROIS,
    TEXT,
)

logger = logging.getLogger(AUTOMM)


class MultiModalFeaturePreprocessor(TransformerMixin, BaseEstimator):
    """
    Preprocess one multimodal pd.DataFrame including image paths, texts, numerical features,
    and categorical features. Each modality may have multiple columns.
    The preprocessor is designed to output model-agnostic features.
    """

    def __init__(
        self,
        config: DictConfig,
        column_types: Dict,
        label_column: Optional[str] = None,
        label_generator: Optional[object] = None,
    ):
        """
        Parameters
        ----------
        config
            Configurations regarding data preprocessing.
        column_types
            The mappings from pd.DataFrame's column names to modality types, e.g., image paths and text.
        label_column
            Name of the label column in pd.DataFrame. Can be None to support zero-short learning.
        label_generator
            A sklearn LabelEncoder instance, or a customized encoder, e.g. NerPreprocessor.
        """
        self._column_types = column_types
        self._label_column = label_column
        self._config = config
        self._feature_generators = dict()

        if label_column:
            if label_generator is None:
                self._label_generator = LabelEncoder()
            else:
                self._label_generator = label_generator

            # Scaler used for numerical labels
            numerical_label_preprocessing = OmegaConf.select(config, "label.numerical_label_preprocessing")
            if numerical_label_preprocessing == "minmaxscaler":
                self._label_scaler = MinMaxScaler()
            elif numerical_label_preprocessing == "standardscaler":
                self._label_scaler = StandardScaler()
            elif numerical_label_preprocessing is None or numerical_label_preprocessing.lower() == "none":
                self._label_scaler = StandardScaler(with_mean=False, with_std=False)
            else:
                raise ValueError(
                    f"The numerical_label_preprocessing={numerical_label_preprocessing} is currently not supported"
                )

        for col_name, col_type in self._column_types.items():
            if col_name == self._label_column:
                continue
            if col_type.startswith((TEXT, IMAGE, ROIS)) or col_type == NULL:
                continue
            elif col_type == CATEGORICAL:
                generator = CategoryFeatureGenerator(
                    cat_order="count",
                    minimum_cat_count=config.categorical.minimum_cat_count,
                    maximum_num_cat=config.categorical.maximum_num_cat,
                    verbosity=0,
                )
                self._feature_generators[col_name] = generator
            elif col_type == NUMERICAL:
                generator = Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        (
                            "scaler",
                            StandardScaler(
                                with_mean=config.numerical.scaler_with_mean,
                                with_std=config.numerical.scaler_with_std,
                            ),
                        ),
                    ]
                )
                self._feature_generators[col_name] = generator

            else:
                raise NotImplementedError(
                    f"Type of the column is not supported currently. Received {col_name}={col_type}."
                )

        self._fit_called = False
        self._fit_x_called = False
        self._fit_y_called = False

        # Some columns will be ignored
        self._ignore_columns_set = set()
        self._text_feature_names = []
        self._categorical_feature_names = []
        self._categorical_num_categories = []
        self._numerical_feature_names = []
        self._image_feature_names = []

    @property
    def label_column(self):
        return self._label_column

    @property
    def column_types(self):
        return self._column_types

    @property
    def image_path_names(self):
        if hasattr(self, "_image_path_names"):
            return self._image_path_names
        else:
            return [col_name for col_name in self._image_feature_names if self._column_types[col_name] == IMAGE_PATH]

    @property
    def image_feature_names(self):
        return self._image_path_names if hasattr(self, "_image_path_names") else self._image_feature_names

    @property
    def text_feature_names(self):
        return self._text_feature_names

    @property
    def categorical_feature_names(self):
        return self._categorical_feature_names

    @property
    def numerical_feature_names(self):
        return self._numerical_feature_names

    @property
    def required_feature_names(self):

        image_feature_names = (
            self._image_path_names if hasattr(self, "_image_path_names") else self._image_feature_names
        )

        return (
            image_feature_names
            + self._text_feature_names
            + self._numerical_feature_names
            + self._categorical_feature_names
        )

    @property
    def all_column_names(self):
        return self._column_types.keys()

    @property
    def categorical_num_categories(self):
        """We will always include the unknown category"""
        return self._categorical_num_categories

    @property
    def config(self):
        return self._config

    @property
    def label_type(self):
        return self._column_types[self._label_column]

    @property
    def label_scaler(self):
        return self._label_scaler

    @property
    def label_generator(self):
        return self._label_generator

    @property
    def fit_called(self):
        return self._fit_x_called and self._fit_y_called

    @property
    def fit_x_called(self):
        return self._fit_x_called

    @property
    def fit_y_called(self):
        return self._fit_y_called

    def get_column_names(self, modality: str):
        if modality.startswith(IMAGE) or modality == ROIS:
            return self._image_path_names if hasattr(self, "_image_path_names") else self._image_feature_names
        elif modality.startswith(TEXT):
            return self._text_feature_names
        elif modality == CATEGORICAL:
            return self._categorical_feature_names
        elif modality == NUMERICAL:
            return self._numerical_feature_names
        elif modality == LABEL:
            return [self._label_column]  # as a list to be consistent with others
        elif self.label_type == NER_ANNOTATION:
            return self._text_feature_names + [self._label_column]
        else:
            raise ValueError(f"Unknown modality: {modality}.")

    def _fit_x(self, X: pd.DataFrame):
        """
        Fit the pd.DataFrame by grouping column names by their modality types. For example, all the
        names of text columns will be put in a list. The CategoryFeatureGenerator, SimpleImputer, and
        StandardScaler will also be initialized.

        Parameters
        ----------
        X
            A multimodal pd.DataFrame.
        """

        if self._fit_x_called:
            raise RuntimeError("fit_x() has been called. Please create a new preprocessor and call it again!")
        self._fit_x_called = True

        for col_name in sorted(X.columns):
            # Just in case X accidentally contains the label column
            if col_name == self._label_column:
                continue
            col_type = self._column_types[col_name]
            logger.debug(f'Process col "{col_name}" with type "{col_type}"')
            col_value = X[col_name]
            if col_type == NULL:
                self._ignore_columns_set.add(col_name)
            elif col_type.startswith(TEXT):
                self._text_feature_names.append(col_name)
            elif col_type == CATEGORICAL:
                if self._config.categorical.convert_to_text:
                    # Convert categorical column as text column
                    col_value = col_value.astype("object")
                    processed_data = col_value.apply(lambda ele: "" if pd.isnull(ele) else str(ele))
                    if len(processed_data.unique()) == 1:
                        self._ignore_columns_set.add(col_name)
                        continue
                    self._text_feature_names.append(col_name)
                else:
                    processed_data = col_value.astype("category")
                    generator = self._feature_generators[col_name]
                    processed_data = generator.fit_transform(pd.DataFrame({col_name: processed_data}))[
                        col_name
                    ].cat.codes.to_numpy(np.int32, copy=True)
                    if len(np.unique(processed_data)) == 1:
                        self._ignore_columns_set.add(col_name)
                        continue
                    num_categories = len(generator.category_map[col_name])
                    # Add one unknown category
                    self._categorical_num_categories.append(num_categories + 1)
                    self._categorical_feature_names.append(col_name)
            elif col_type == NUMERICAL:
                processed_data = pd.to_numeric(col_value)
                if len(processed_data.unique()) == 1:
                    self._ignore_columns_set.add(col_name)
                    continue
                if self._config.numerical.convert_to_text:
                    self._text_feature_names.append(col_name)
                else:
                    generator = self._feature_generators[col_name]
                    generator.fit(np.expand_dims(processed_data.to_numpy(), axis=-1))
                    self._numerical_feature_names.append(col_name)
            elif col_type.startswith(IMAGE) or col_type == ROIS:  # TODO: Use transform_multimodal and remove this hack
                self._image_feature_names.append(col_name)
            else:
                raise NotImplementedError(
                    f"Type of the column is not supported currently. Received {col_name}={col_type}."
                )

    def _fit_y(self, y: pd.Series, X: Optional[pd.DataFrame] = None):
        """
        Fit the label column data to initialize the label encoder or scalar.

        Parameters
        ----------
        y
            The Label column data.
        """
        if self._fit_y_called:
            raise RuntimeError("fit_y() has been called. Please create a new preprocessor and call it again!")
        self._fit_y_called = True
        logger.debug(f'Process col "{self._label_column}" with type label')
        if self.label_type == CATEGORICAL:
            self._label_generator.fit(y)
        elif self.label_type == NUMERICAL:
            y = pd.to_numeric(y).to_numpy()
            self._label_scaler.fit(np.expand_dims(y, axis=-1))
        elif self.label_type == ROIS:
            pass  # Do nothing. TODO: Shall we call fit here?
        elif self.label_type == NER_ANNOTATION:
            self._label_generator.fit(y, X[self._text_feature_names[0]])
        else:
            raise NotImplementedError(f"Type of label column is not supported. Label column type={self.label_type}")

    def fit(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None):
        """
        Fit the dataframe preprocessor with features X and labels y.

        Parameters
        ----------
        X
            The multimodal features in the format of pd.DataFrame.
        y
            The Label data in the format of pd.Series.
        """
        if X is not None:
            self._fit_x(X=X)
        if y is not None:
            self._fit_y(y=y, X=X)

    def transform_text(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Preprocess text data by collecting them together. May need to format
        the categorical and numerical data into strings if using them so.
        This function needs to be called preceding the text processor in "process_text.py".

        Parameters
        ----------
        df
            The multimodal pd.DataFrame.

        Returns
        -------
        text_features
            All the text data stored in a dictionary.
        text_types
            The column types of these text data, e.g., text or text_identifier.
        """
        assert (
            self._fit_called or self._fit_x_called
        ), "You will need to first call preprocessor.fit_x() before calling preprocessor.transform_text."
        text_features = {}
        text_types = {}
        for col_name in self._text_feature_names:
            col_value = df[col_name]
            col_type = self._column_types[col_name]
            if col_type == TEXT or col_type == CATEGORICAL:
                # TODO: do we need to consider whether categorical values are valid text?
                col_value = col_value.astype("object")
                processed_data = col_value.apply(lambda ele: "" if pd.isnull(ele) else str(ele))
            elif col_type == NUMERICAL:
                processed_data = pd.to_numeric(col_value).apply("{:.3f}".format)
            elif col_type == f"{TEXT}_{IDENTIFIER}":
                processed_data = col_value
            else:
                raise ValueError(f"Column {col_name} has type {col_type}, which can't be converted to text.")

            text_features[col_name] = processed_data.values.tolist()
            text_types[col_name] = col_type

        return text_features, text_types

    def transform_image(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, List[List[str]]], Dict[str, str]]:
        """
        Preprocess image data by collecting their paths together. If one sample has multiple images
        in an image column, assume that their image paths are separated by ";".
        This function needs to be called preceding the image processor in "process_image.py".

        Parameters
        ----------
        df
            The multimodal pd.DataFrame.

        Returns
        -------
        image_features
            All the image data stored in a dictionary.
        image_types
            The column types of these image data, e.g., image_path or image_identifier.
        """
        assert (
            self._fit_called or self._fit_x_called
        ), "You will need to first call preprocessor.fit_x() before calling preprocessor.transform_image."

        image_features = {}
        image_types = {}
        for col_name in self._image_feature_names:
            col_value = df[col_name]
            col_type = self._column_types[col_name]

            if col_type == ROIS:
                processed_data = df[col_name].tolist()
            elif col_type == IMAGE_PATH or IMAGE:
                processed_data = col_value.apply(lambda ele: ele.split(";")).tolist()
            elif col_type == f"{IMAGE}_{IDENTIFIER}":
                processed_data = col_value
            else:
                raise ValueError(f"Unknown image type {col_type} for column {col_name}")

            image_features[col_name] = processed_data
            image_types[col_name] = self._column_types[col_name]

        return image_features, image_types

    def transform_numerical(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, NDArray[(Any,), np.float32]], None]:
        """
        Preprocess numerical data by using SimpleImputer to fill possible missing values
        and StandardScaler to standardize the values (z = (x - mean) / std).
        This function needs to be called preceding the numerical processor in "process_numerical.py".

        Parameters
        ----------
        df
            The multimodal pd.DataFrame.

        Returns
        -------
        numerical_features
            All the numerical features (a dictionary of np.ndarray).
        None
            The column types of numerical data, which is None currently since only one numerical type exists.
        """
        assert (
            self._fit_called or self._fit_x_called
        ), "You will need to first call preprocessor.fit before calling preprocessor.transform_numerical."
        numerical_features = {}
        for col_name in self._numerical_feature_names:
            generator = self._feature_generators[col_name]
            col_value = pd.to_numeric(df[col_name]).to_numpy()
            processed_data = generator.transform(np.expand_dims(col_value, axis=-1))[:, 0]
            numerical_features[col_name] = processed_data.astype(np.float32)

        return numerical_features, None

    def transform_categorical(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, NDArray[(Any,), np.int32]], None]:
        """
        Preprocess categorical data by using CategoryFeatureGenerator to generate
        categorical encodings, i.e., integers. This function needs to be called
        preceding the categorical processor in "process_categorical.py".

        Parameters
        ----------
        df
            The multimodal pd.DataFrame.

        Returns
        -------
        categorical_features
            All the categorical encodings (a dictionary of np.ndarray).
        None
            The column types of categorical data, which is None currently since only one categorical type exists.
        """
        assert (
            self._fit_called or self._fit_x_called
        ), "You will need to first call preprocessor.fit before calling preprocessor.transform_categorical."
        categorical_features = {}
        for col_name, num_category in zip(self._categorical_feature_names, self._categorical_num_categories):
            col_value = df[col_name]
            processed_data = col_value.astype("category")
            generator = self._feature_generators[col_name]
            processed_data = generator.transform(pd.DataFrame({col_name: processed_data}))[
                col_name
            ].cat.codes.to_numpy(np.int32, copy=True)
            processed_data[processed_data < 0] = num_category - 1
            categorical_features[col_name] = processed_data

        return categorical_features, None

    def transform_label(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, NDArray[(Any,), Any]], Dict[str, str]]:
        """
        Preprocess ground-truth labels by using LabelEncoder to generate class labels for
        classification tasks or using StandardScaler to standardize numerical values
        (z = (x - mean) / std) for regression tasks. This function needs to be called
        preceding the label processor in "process_label.py".

        Parameters
        ----------
        df
            The multimodal pd.DataFrame.

        Returns
        -------
        labels
            All the labels (a dictionary of np.ndarray).
        label_types
            The label column types.
        """
        assert (
            self._fit_called or self._fit_y_called
        ), "You will need to first call preprocessor.fit_y() before calling preprocessor.transform_label."
        y_df = df[self._label_column]
        if self.label_type == CATEGORICAL:
            y = self._label_generator.transform(y_df)
        elif self.label_type == NUMERICAL:
            y = pd.to_numeric(y_df).to_numpy()
            y = self._label_scaler.transform(np.expand_dims(y, axis=-1))[:, 0].astype(np.float32)
        elif self.label_type == ROIS:
            y = y_df.to_list()
        elif self.label_type == NER_ANNOTATION:
            y = self._label_generator.transform(y_df)
        else:
            raise NotImplementedError

        return {self._label_column: y}, {self._label_column: self.label_type}

    def transform_ner(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, NDArray[(Any,), Any]], Dict[str, str]]:
        ret_data, ret_type = {}, {}
        if self.label_type == NER_ANNOTATION:
            x = self.transform_text(df)
            ret_data.update(x[0])
            ret_type.update(x[1])
            if self._label_column in df:
                y = self.transform_label(df)
                ret_data.update(y[0])
                ret_type.update(y[1])
        else:
            raise NotImplementedError

        return ret_data, ret_type

    def transform_label_for_metric(
        self,
        df: pd.DataFrame,
        tokenizer: Optional[Any] = None,
    ) -> NDArray[(Any,), Any]:
        """
        Prepare ground-truth labels to compute metric scores in evaluation. Note that
        numerical values are not normalized since we want to evaluate the model performance
        on the raw numerical values.

        Parameters
        ----------
        df
            The multimodal pd.DataFrame for evaluation.

        Returns
        -------
        Ground-truth labels ready to compute metric scores.
        """
        assert (
            self._fit_called or self._fit_y_called
        ), "You will need to first call preprocessor.fit_y() before calling preprocessor.transform_label_for_metric."
        y_df = df[self._label_column]
        if self.label_type == CATEGORICAL:
            # need to encode to integer labels
            y = self._label_generator.transform(y_df)
        elif self.label_type == NUMERICAL:
            # need to compute the metric on the raw numerical values (no normalization)
            y = pd.to_numeric(y_df).to_numpy()
        elif self.label_type == NER_ANNOTATION:
            x_df = df[self._text_feature_names[0]]
            y = self._label_generator.transform_label_for_metric(y_df, x_df, tokenizer)
        else:
            raise NotImplementedError

        return y

    def transform_prediction(
        self,
        y_pred: Union[np.ndarray, dict],
        inverse_categorical: bool = True,
    ) -> NDArray[(Any,), Any]:
        """
        Transform model's output logits/probability into class labels for classification
        or raw numerical values for regression.

        Parameters
        ----------
        y_pred
            The model's output logits/probability.
        inverse_categorical
            Whether to transform categorical value back to the original space, e.g., string values.
        loss_func
            The loss function of the model.

        Returns
        -------
        Predicted labels ready to compute metric scores.
        """
        assert (
            self._fit_called or self._fit_y_called
        ), "You will need to first call preprocessor.fit_y() before calling preprocessor.transform_prediction."

        if self.label_type == CATEGORICAL:
            assert y_pred.shape[1] >= 2
            y_pred = y_pred.argmax(axis=1)
            # Transform the predicted label back to the original space (e.g., string values)
            if inverse_categorical:
                y_pred = self._label_generator.inverse_transform(y_pred)
        elif self.label_type == NUMERICAL:
            y_pred = self._label_scaler.inverse_transform(y_pred)
            y_pred = np.squeeze(y_pred)
            # Convert nan to 0
            y_pred = np.nan_to_num(y_pred)
        elif self.label_type == NER_ANNOTATION:
            y_pred = self._label_generator.inverse_transform(y_pred)
            if inverse_categorical:
                # Return annotations and offsets
                y_pred = y_pred[1]
            else:
                y_pred = y_pred[0]
        else:
            raise NotImplementedError

        return y_pred
