import logging
import pandas as pd
import numpy as np
import torch
import collections
from typing import Callable, Iterator, Union, Optional, List, Any, Dict
from nptyping import NDArray
from autogluon.features import CategoryFeatureGenerator
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
)
from sklearn.base import (
    TransformerMixin,
    BaseEstimator,
)
from ..constants import (
    CATEGORICAL, NUMERICAL, TEXT,
    IMAGE_PATH, NULL, AUTOMM
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
            config: dict,
            column_types: collections.OrderedDict,
            label_column: str,
            label_generator: Optional[LabelEncoder] = None,
    ):
        """
        Parameters
        ----------
        config
            Configurations regarding data preprocessing.
        column_types
            The mappings from pd.DataFrame's column names to modality types, e.g., image paths and text.
        label_column
            Name of the label column in pd.DataFrame.
        label_generator
            A sklearn LabelEncoder instance.
        """
        self._column_types = column_types
        self._label_column = label_column
        self._config = config
        self._feature_generators = dict()
        if label_generator is None:
            self._label_generator = LabelEncoder()
        else:
            self._label_generator = label_generator
        self._label_scaler = StandardScaler()  # Scaler used for numerical labels
        for col_name, col_type in self._column_types.items():
            if col_name == self._label_column:
                continue
            if col_type == TEXT:
                continue
            elif col_type == CATEGORICAL:
                generator = CategoryFeatureGenerator(
                    cat_order='count',
                    minimum_cat_count=config["categorical"]["minimum_cat_count"],
                    maximum_num_cat=config["categorical"]["maximum_num_cat"],
                    verbosity=0)
                self._feature_generators[col_name] = generator
            elif col_type == NUMERICAL:
                generator = Pipeline(
                    [('imputer', SimpleImputer()),
                     ('scaler', StandardScaler(with_mean=config["numerical"]["scaler_with_mean"],
                                               with_std=config["numerical"]["scaler_with_std"]))]
                )
                self._feature_generators[col_name] = generator

        self._fit_called = False

        # Some columns will be ignored
        self._ignore_columns_set = set()
        self._text_feature_names = []
        self._categorical_feature_names = []
        self._categorical_num_categories = []
        self._numerical_feature_names = []
        self._image_path_names = []

    @property
    def label_column(self):
        return self._label_column

    @property
    def image_path_names(self):
        return self._image_path_names

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
        return self._fit_called

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
    ):
        """
        Fit the pd.DataFrame by grouping column names by their modality types. For example, all the
        names of text columns will be put in a list. The CategoryFeatureGenerator, SimpleImputer,
        StandardScaler, and LabelEncoder will also be initialized.

        Parameters
        ----------
        X
            The multimodal pd.DataFrame.
        y
            The label pd.Series.
        """
        if self._fit_called:
            raise RuntimeError(
                "Fit has been called. Please create a new preprocessor and call fit again!"
            )
        self._fit_called = True
        for col_name in sorted(X.columns):
            # Just in case X accidentally contains the label column
            if col_name == self._label_column:
                continue
            col_type = self._column_types[col_name]
            logger.debug(f'Process col "{col_name}" with type "{col_type}"')
            col_value = X[col_name]
            if col_type == NULL:
                self._ignore_columns_set.add(col_name)
                continue
            elif col_type == TEXT:
                self._text_feature_names.append(col_name)
            elif col_type == CATEGORICAL:
                if self._config["categorical"]["convert_to_text"]:
                    # Convert categorical column as text column
                    col_value = col_value.astype("object")
                    processed_data = col_value.apply(lambda ele: '' if pd.isnull(ele) else str(ele))
                    if len(processed_data.unique()) == 1:
                        self._ignore_columns_set.add(col_name)
                        continue
                    self._text_feature_names.append(col_name)
                else:
                    processed_data = col_value.astype('category')
                    generator = self._feature_generators[col_name]
                    processed_data = generator.fit_transform(
                        pd.DataFrame({col_name: processed_data}))[col_name] \
                        .cat.codes.to_numpy(np.int32, copy=True)
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
                if self._config["numerical"]["convert_to_text"]:
                    self._text_feature_names.append(col_name)
                else:
                    generator = self._feature_generators[col_name]
                    generator.fit(np.expand_dims(processed_data.to_numpy(), axis=-1))
                    self._numerical_feature_names.append(col_name)
            elif col_type == IMAGE_PATH:
                self._image_path_names.append(col_name)
            else:
                raise NotImplementedError(
                    f"Type of the column is not supported currently. "
                    f"Received {col_name}={col_type}."
                )
        if self.label_type == CATEGORICAL:
            self._label_generator.fit(y)
        elif self.label_type == NUMERICAL:
            y = pd.to_numeric(y).to_numpy()
            self._label_scaler.fit(np.expand_dims(y, axis=-1))
        else:
            raise NotImplementedError(
                f"Type of label column is not supported. "
                f"Label column type={self._label_column}"
            )

    def transform_text(
            self,
            df: pd.DataFrame,
    ) -> Dict[str, List[str]]:
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
        All the text data stored in a dictionary.
        """
        assert self._fit_called, 'You will need to first call ' \
                                 'preprocessor.fit before calling ' \
                                 'preprocessor.transform.'
        text_features = {}
        for col_name in self._text_feature_names:
            col_value = df[col_name]
            col_type = self._column_types[col_name]
            if col_type == TEXT or col_type == CATEGORICAL:
                # TODO: do we need to consider whether categorical values are valid text?
                col_value = col_value.astype("object")
                processed_data = col_value.apply(lambda ele: '' if pd.isnull(ele) else str(ele))
            elif col_type == NUMERICAL:
                processed_data = pd.to_numeric(col_value).apply('{:.3f}'.format)
            else:
                raise NotImplementedError

            text_features[col_name] = processed_data.values.tolist()

        return text_features

    def transform_image(
            self,
            df: pd.DataFrame,
    ) -> Dict[str, List[List[str]]]:
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
        All the image paths stored in a dictionary.
        """
        assert self._fit_called, 'You will need to first call ' \
                                 'preprocessor.fit before calling ' \
                                 'preprocessor.transform.'
        image_paths = {}
        for col_name in self._image_path_names:
            processed_data = df[col_name].apply(lambda ele: ele.split(';')).tolist()
            image_paths[col_name] = processed_data
        return image_paths

    def transform_numerical(
            self,
            df: pd.DataFrame,
    ) -> Dict[str, NDArray[(Any,), np.float32]]:
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
        All the numerical features (a dictionary of np.ndarray).
        """
        assert self._fit_called, 'You will need to first call ' \
                                 'preprocessor.fit before calling ' \
                                 'preprocessor.transform.'
        numerical_features = {}
        for col_name in self._numerical_feature_names:
            generator = self._feature_generators[col_name]
            col_value = pd.to_numeric(df[col_name]).to_numpy()
            processed_data = generator.transform(np.expand_dims(col_value, axis=-1))[:, 0]
            numerical_features[col_name] = processed_data.astype(np.float32)

        return numerical_features

    def transform_categorical(
            self,
            df: pd.DataFrame,
    ) -> Dict[str, NDArray[(Any,), np.int32]]:
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
        All the categorical encodings (a dictionary of np.ndarray).
        """
        assert self._fit_called, 'You will need to first call ' \
                                 'preprocessor.fit before calling ' \
                                 'preprocessor.transform.'
        categorical_features = {}
        for col_name, num_category in zip(self._categorical_feature_names,
                                          self._categorical_num_categories):
            col_value = df[col_name]
            processed_data = col_value.astype('category')
            generator = self._feature_generators[col_name]
            processed_data = generator.transform(
                pd.DataFrame({col_name: processed_data}))[col_name] \
                .cat.codes.to_numpy(np.int32, copy=True)
            processed_data[processed_data < 0] = num_category - 1
            categorical_features[col_name] = processed_data

        return categorical_features

    def transform_label(
            self,
            df: pd.DataFrame,
    ) -> Dict[str, NDArray[(Any,), Any]]:
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
        All the labels (a dictionary of np.ndarray).
        """
        assert self._fit_called, 'You will need to first call ' \
                                 'preprocessor.fit before calling ' \
                                 'preprocessor.transform.'
        y_df = df[self._label_column]
        if self.label_type == CATEGORICAL:
            y = self._label_generator.transform(y_df)
        elif self.label_type == NUMERICAL:
            y = pd.to_numeric(y_df).to_numpy()
            y = self._label_scaler.transform(np.expand_dims(y, axis=-1))[:, 0].astype(np.float32)
        else:
            raise NotImplementedError

        return {self._label_column: y}

    def transform_label_for_metric(
            self,
            df: pd.DataFrame,
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
        assert self._fit_called, 'You will need to first call ' \
                                 'preprocessor.fit before calling ' \
                                 'preprocessor.transform.'
        y_df = df[self._label_column]
        if self.label_type == CATEGORICAL:
            # need to encode to integer labels
            y = self._label_generator.transform(y_df)
        elif self.label_type == NUMERICAL:
            # need to compute the metric on the raw numerical values (no normalization)
            y = pd.to_numeric(y_df).to_numpy()
        else:
            raise NotImplementedError

        return y

    def transform_prediction(
            self,
            y_pred: torch.Tensor,
            inverse_categorical: bool = True,
    ) -> NDArray[(Any,), Any]:
        """
        Transform model's output logits into class labels for classification
        or raw numerical values for regression.

        Parameters
        ----------
        y_pred
            The model's output logits.
        inverse_categorical
            Whether to transform categorical value back to the original space, e.g., string values.

        Returns
        -------
        Predicted labels ready to compute metric scores.
        """
        y_pred = y_pred.detach().cpu().float().numpy()

        if self.label_type == CATEGORICAL:
            assert y_pred.shape[1] >= 2
            y_pred = y_pred.argmax(axis=1)
            # Transform the predicted label back to the original space (e.g., string values)
            if inverse_categorical:
                y_pred = self._label_generator.inverse_transform(y_pred)
        elif self.label_type == NUMERICAL:
            y_pred = self._label_scaler.inverse_transform(y_pred)
            y_pred = np.squeeze(y_pred)
        else:
            raise NotImplementedError

        return y_pred
