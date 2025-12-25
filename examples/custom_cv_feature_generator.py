"""
Custom CV-Aware Feature Generator for AutoGluon

This module demonstrates how to integrate a custom feature generator that uses
target encoding (or any label-dependent feature engineering) with AutoGluon's
cross-validation pipeline without data leakage.

The key insight is that AutoGluon applies feature generators ONCE before CV splitting.
To avoid data leakage, your feature generator must perform its own internal CV splits
during fit_transform, computing OOF (out-of-fold) encodings for training data.

Author: AutoGluon Integration Example
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from sklearn.model_selection import KFold, StratifiedKFold
import copy

from autogluon.common.utils.cv_splitter import CVSplitter
from autogluon.features.generators.abstract import AbstractFeatureGenerator


class CVAwareFeatureGenerator(AbstractFeatureGenerator):
    """
    A wrapper that makes any fit_transform/transform-style feature generator
    CV-aware to prevent data leakage when using target encoding features.

    This is the recommended pattern for integrating custom feature generators
    that use label information (like target encoding from genetic programming).

    Parameters
    ----------
    base_generator : object
        Your custom feature generator that implements:
        - fit_transform(X, y) -> X_transformed
        - transform(X) -> X_transformed
        This could be your GP-based feature generator.
    n_splits : int, default=5
        Number of folds for internal CV (should match AutoGluon's k_fold setting).
    stratify : bool, default=True
        Whether to use stratified splits for classification.
    random_state : int, default=42
        Random state for reproducibility.
    keep_original : bool, default=False
        Whether to keep original features alongside generated ones.
    **kwargs
        Additional arguments passed to AbstractFeatureGenerator.

    Example
    -------
    ```python
    from autogluon.tabular import TabularPredictor

    # Your custom GP-based feature generator
    class GPFeatureGenerator:
        def fit_transform(self, X, y):
            # Your GP feature engineering logic here
            # This might include target encoding, aggregations, etc.
            return X_with_new_features

        def transform(self, X):
            # Apply fitted transformations
            return X_transformed

    # Wrap it for CV-awareness
    cv_aware_generator = CVAwareFeatureGenerator(
        base_generator=GPFeatureGenerator(),
        n_splits=5,  # Match AutoGluon's k_fold
    )

    # Use with AutoGluon
    from autogluon.features.generators import PipelineFeatureGenerator

    custom_pipeline = PipelineFeatureGenerator(
        generators=[cv_aware_generator],
        post_generators=[]
    )

    predictor = TabularPredictor(label='target')
    predictor.fit(train_data, feature_generator=custom_pipeline)
    ```
    """

    def __init__(
        self,
        base_generator: Any,
        n_splits: int = 5,
        stratify: bool = True,
        random_state: int = 42,
        keep_original: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_generator = base_generator
        self.n_splits = n_splits
        self.stratify = stratify
        self.random_state = random_state
        self.keep_original = keep_original

        # Will be set during fit
        self._full_data_generator = None  # For test-time transform
        self._original_cols = None
        self._generated_cols = None

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        Fit the feature generator using internal CV to prevent data leakage.

        Strategy:
        1. Split data into K folds internally
        2. For each fold: fit generator on training fold, transform validation fold
        3. Concatenate OOF predictions to get leak-free training features
        4. Also fit on full data for test-time inference
        """
        self._original_cols = list(X.columns)
        n = len(X)
        original_index = X.index

        # Reset index for consistent indexing
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # Create CV splitter matching AutoGluon's pattern
        if self.stratify:
            splitter_cls = StratifiedKFold
        else:
            splitter_cls = KFold

        cv_splitter = CVSplitter(
            splitter_cls=splitter_cls,
            n_splits=self.n_splits,
            random_state=self.random_state,
            stratify=self.stratify,
            shuffle=True,
        )

        kf_splits = list(cv_splitter.split(X, y))

        # Initialize OOF container
        # First, do a trial fit to determine output shape
        trial_generator = copy.deepcopy(self.base_generator)
        X_trial = trial_generator.fit_transform(X.iloc[:min(100, n)], y.iloc[:min(100, n)])

        # Identify new columns generated
        if hasattr(X_trial, 'columns'):
            all_trial_cols = set(X_trial.columns)
            original_cols_set = set(self._original_cols)
            self._generated_cols = [c for c in X_trial.columns if c not in original_cols_set]
        else:
            # Assume numpy array output
            n_new_features = X_trial.shape[1] - len(self._original_cols)
            self._generated_cols = [f'gp_feature_{i}' for i in range(n_new_features)]

        n_generated = len(self._generated_cols)

        # OOF array for generated features only
        oof_generated = np.zeros((n, n_generated), dtype=np.float64)

        # Compute OOF encodings
        for fold_idx, (train_idx, val_idx) in enumerate(kf_splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            # Fit generator on training fold only
            fold_generator = copy.deepcopy(self.base_generator)
            fold_generator.fit_transform(X_train, y_train)

            # Transform validation fold (OOF)
            X_val_transformed = fold_generator.transform(X_val)

            # Extract only the generated features
            if hasattr(X_val_transformed, 'values'):
                # DataFrame
                generated_values = X_val_transformed[self._generated_cols].values
            else:
                # Numpy array - assume generated features are appended
                generated_values = X_val_transformed[:, len(self._original_cols):]

            oof_generated[val_idx] = generated_values

        # Fit on full data for test-time inference
        self._full_data_generator = copy.deepcopy(self.base_generator)
        self._full_data_generator.fit_transform(X, y)

        # Build output DataFrame with OOF encodings
        oof_df = pd.DataFrame(
            oof_generated,
            columns=self._generated_cols,
            index=original_index
        )

        if self.keep_original:
            # Restore original index
            X.index = original_index
            X_out = pd.concat([X, oof_df], axis=1)
        else:
            X_out = oof_df

        return X_out, dict()

    def _transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform new data using the generator fitted on full training data.
        """
        if self._full_data_generator is None:
            raise RuntimeError("Feature generator must be fit before transform")

        X_transformed = self._full_data_generator.transform(X)

        # Extract generated features
        if hasattr(X_transformed, 'columns'):
            result = X_transformed[self._generated_cols]
        else:
            result = pd.DataFrame(
                X_transformed[:, len(self._original_cols):],
                columns=self._generated_cols,
                index=X.index
            )

        if self.keep_original:
            result = pd.concat([X[self._original_cols], result], axis=1)

        return result

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()


class SynchronizedCVFeatureGenerator(AbstractFeatureGenerator):
    """
    An alternative approach that synchronizes internal CV splits with AutoGluon's CV.

    This generator accepts fold indices at fit time, allowing it to be used
    in conjunction with a modified training loop that passes fold information.

    Use this when you need exact synchronization between feature generation CV
    and model training CV (e.g., for stacking or when fold consistency is critical).

    Parameters
    ----------
    base_generator : object
        Your custom feature generator with fit_transform/transform API.
    **kwargs
        Additional arguments passed to AbstractFeatureGenerator.

    Notes
    -----
    This approach requires modifications to how AutoGluon calls the feature generator.
    See the FoldAwareBaggedEnsembleModel example for integration details.
    """

    def __init__(self, base_generator: Any, **kwargs):
        super().__init__(**kwargs)
        self.base_generator = base_generator
        self._fold_generators = {}  # fold_id -> fitted generator
        self._full_generator = None
        self._original_cols = None
        self._generated_cols = None

    def fit_for_fold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fold_id: int,
        train_idx: np.ndarray,
        **kwargs
    ) -> None:
        """
        Fit the generator for a specific fold using only training indices.

        Parameters
        ----------
        X : pd.DataFrame
            Full training data
        y : pd.Series
            Full training labels
        fold_id : int
            Identifier for this fold
        train_idx : np.ndarray
            Indices of training samples for this fold
        """
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        fold_generator = copy.deepcopy(self.base_generator)
        fold_generator.fit_transform(X_train, y_train)
        self._fold_generators[fold_id] = fold_generator

    def transform_for_fold(
        self,
        X: pd.DataFrame,
        fold_id: int,
    ) -> pd.DataFrame:
        """
        Transform data using the generator fitted for a specific fold.
        """
        if fold_id not in self._fold_generators:
            raise ValueError(f"No generator fitted for fold {fold_id}")

        return self._fold_generators[fold_id].transform(X)

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        Default fit_transform using full data (for test-time inference setup).
        """
        self._original_cols = list(X.columns)

        self._full_generator = copy.deepcopy(self.base_generator)
        X_out = self._full_generator.fit_transform(X, y)

        if hasattr(X_out, 'columns'):
            original_set = set(self._original_cols)
            self._generated_cols = [c for c in X_out.columns if c not in original_set]

        return X_out, dict()

    def _transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform using full-data fitted generator."""
        if self._full_generator is None:
            raise RuntimeError("Generator not fitted")
        return self._full_generator.transform(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()


# ==============================================================================
# Option 2: Custom Model with Internal Feature Generation
# ==============================================================================

class FeatureGeneratingModelMixin:
    """
    Mixin for AutoGluon models that need to apply custom feature generation
    per-fold during training.

    This approach moves feature generation INTO the model, so it naturally
    receives fold-specific training data without requiring modifications
    to the bagging infrastructure.

    Example
    -------
    ```python
    from autogluon.core.models import AbstractModel

    class GPFeatureModel(FeatureGeneratingModelMixin, AbstractModel):
        def __init__(self, feature_generator, **kwargs):
            super().__init__(**kwargs)
            self.feature_generator = feature_generator
            self._fitted_generator = None

        def _fit(self, X, y, X_val=None, y_val=None, **kwargs):
            # Feature generator is fitted on training fold only
            self._fitted_generator = copy.deepcopy(self.feature_generator)
            X_transformed = self._fitted_generator.fit_transform(X, y)

            if X_val is not None:
                X_val_transformed = self._fitted_generator.transform(X_val)

            # Train underlying model on transformed features
            super()._fit(X_transformed, y, X_val_transformed, y_val, **kwargs)

        def _predict_proba(self, X, **kwargs):
            X_transformed = self._fitted_generator.transform(X)
            return super()._predict_proba(X_transformed, **kwargs)
    ```
    """
    pass


# ==============================================================================
# Full Integration Example
# ==============================================================================

def example_usage():
    """
    Complete example showing how to integrate a custom GP feature generator
    with AutoGluon's cross-validation.
    """
    # Simulated GP-based feature generator
    class SimpleGPFeatureGenerator:
        """
        Example of a simple feature generator that uses target encoding.
        Replace this with your actual GP-based generator.
        """
        def __init__(self):
            self.encodings_ = {}
            self.global_mean_ = None

        def fit_transform(self, X, y):
            X = X.copy()
            self.global_mean_ = y.mean()

            # Target encode categorical columns
            for col in X.select_dtypes(include=['object', 'category']).columns:
                encoding = y.groupby(X[col]).mean()
                self.encodings_[col] = encoding
                X[f'{col}_te'] = X[col].map(encoding).fillna(self.global_mean_)

            return X

        def transform(self, X):
            X = X.copy()
            for col, encoding in self.encodings_.items():
                X[f'{col}_te'] = X[col].map(encoding).fillna(self.global_mean_)
            return X

    # Wrap with CV-aware generator
    cv_generator = CVAwareFeatureGenerator(
        base_generator=SimpleGPFeatureGenerator(),
        n_splits=5,
        stratify=True,
        keep_original=True,
    )

    print("CVAwareFeatureGenerator created successfully!")
    print("This generator will:")
    print("1. Perform internal 5-fold CV during fit_transform")
    print("2. Compute OOF encodings to prevent data leakage")
    print("3. Fit on full data for test-time inference")

    return cv_generator


if __name__ == "__main__":
    example_usage()
