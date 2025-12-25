"""
Advanced CV-Synchronized Feature Generator Integration for AutoGluon

This module demonstrates an advanced approach where the feature generator's CV splits
are exactly synchronized with AutoGluon's bagging CV splits. This ensures that:
1. Feature generation CV matches model training CV exactly
2. No data leakage occurs at any point
3. The same random seed produces the same splits

This is useful when:
- You need exact reproducibility between feature generation and model training
- Your feature generator's CV must align with AutoGluon's for stacking
- You want to implement your own training loop with custom feature generation

Author: AutoGluon Integration Example
"""

import copy
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from autogluon.common.utils.cv_splitter import CVSplitter
from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from autogluon.core.models.abstract.abstract_model import AbstractModel

logger = logging.getLogger(__name__)


class FoldAwareTrainer:
    """
    A custom training coordinator that synchronizes feature generation CV
    with AutoGluon model training CV.

    This class:
    1. Creates CV splits ONCE
    2. Passes split information to the feature generator for per-fold fitting
    3. Uses the same splits for model training

    Example
    -------
    ```python
    from autogluon.tabular import TabularPredictor

    trainer = FoldAwareTrainer(
        feature_generator=MyGPFeatureGenerator(),
        n_splits=5,
        random_state=42,
    )

    # Train with synchronized CV
    X_train_transformed, fold_generators = trainer.fit_transform_cv(X_train, y_train)

    # Use transformed data with AutoGluon
    predictor = TabularPredictor(label='target')
    predictor.fit(
        X_train_transformed,
        hyperparameters={'GBM': {}},
        ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'},
    )
    ```
    """

    def __init__(
        self,
        feature_generator: Any,
        n_splits: int = 5,
        n_repeats: int = 1,
        stratify: bool = True,
        random_state: int = 0,
    ):
        """
        Initialize the fold-aware trainer.

        Parameters
        ----------
        feature_generator : Any
            Your custom feature generator with fit_transform/transform API.
        n_splits : int
            Number of CV folds (should match AutoGluon's k_fold).
        n_repeats : int
            Number of bag repeats.
        stratify : bool
            Whether to use stratified splits.
        random_state : int
            Random state for reproducibility.
        """
        self.feature_generator = feature_generator
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.stratify = stratify
        self.random_state = random_state

        self._cv_splitter = None
        self._fold_generators = {}  # (repeat, fold) -> fitted generator
        self._full_generator = None
        self._splits_cache = None

    def _get_cv_splitter(self) -> CVSplitter:
        """Create CV splitter matching AutoGluon's configuration."""
        if self._cv_splitter is None:
            self._cv_splitter = CVSplitter(
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                stratify=self.stratify,
                random_state=self.random_state,
                shuffle=True,
            )
        return self._cv_splitter

    def get_splits(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get CV splits. Cached for consistency.

        Returns list of (train_idx, val_idx) tuples.
        """
        if self._splits_cache is not None:
            return self._splits_cache

        cv_splitter = self._get_cv_splitter()
        self._splits_cache = cv_splitter.split(X, y)
        return self._splits_cache

    def fit_transform_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], Any]]:
        """
        Fit feature generators per fold and return OOF-transformed features.

        This method:
        1. Creates CV splits
        2. For each fold: fits generator on training data, transforms validation data
        3. Concatenates OOF predictions
        4. Also fits on full data for test-time inference

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Training labels.

        Returns
        -------
        X_oof : pd.DataFrame
            OOF-transformed features (no data leakage).
        fold_generators : dict
            Dictionary mapping (repeat, fold) to fitted generators.
        """
        n = len(X)
        original_index = X.index.copy()

        # Reset indices for consistent indexing
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        splits = self.get_splits(X, y)

        # Determine output shape from trial fit
        trial_gen = copy.deepcopy(self.feature_generator)
        X_trial = trial_gen.fit_transform(X.iloc[:min(100, n)], y.iloc[:min(100, n)])
        n_output_cols = X_trial.shape[1] if hasattr(X_trial, 'shape') else len(X_trial.columns)

        # Initialize OOF container
        if hasattr(X_trial, 'columns'):
            oof_columns = list(X_trial.columns)
            oof = pd.DataFrame(
                np.zeros((n, n_output_cols), dtype=np.float64),
                columns=oof_columns
            )
        else:
            oof = np.zeros((n, n_output_cols), dtype=np.float64)

        # Process each fold
        fold_generators = {}
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            repeat = fold_idx // self.n_splits
            fold_in_repeat = fold_idx % self.n_splits

            logger.info(f"Processing fold S{repeat+1}F{fold_in_repeat+1}: "
                       f"train={len(train_idx)}, val={len(val_idx)}")

            # Fit generator on training fold only
            fold_gen = copy.deepcopy(self.feature_generator)
            fold_gen.fit_transform(X.iloc[train_idx], y.iloc[train_idx])

            # Transform validation fold (OOF)
            X_val_transformed = fold_gen.transform(X.iloc[val_idx])

            # Store OOF predictions
            if hasattr(oof, 'iloc'):
                oof.iloc[val_idx] = X_val_transformed.values if hasattr(X_val_transformed, 'values') else X_val_transformed
            else:
                oof[val_idx] = X_val_transformed

            # Store fitted generator
            fold_generators[(repeat, fold_in_repeat)] = fold_gen

        # Fit on full data for inference
        self._full_generator = copy.deepcopy(self.feature_generator)
        self._full_generator.fit_transform(X, y)
        self._fold_generators = fold_generators

        # Restore original index
        if hasattr(oof, 'index'):
            oof.index = original_index

        return oof, fold_generators

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the full-data fitted generator.
        """
        if self._full_generator is None:
            raise RuntimeError("Must call fit_transform_cv first")
        return self._full_generator.transform(X)

    def transform_for_fold(
        self,
        X: pd.DataFrame,
        repeat: int = 0,
        fold: int = 0,
    ) -> pd.DataFrame:
        """
        Transform data using a specific fold's generator.

        Useful for creating predictions with fold-specific feature generators.
        """
        key = (repeat, fold)
        if key not in self._fold_generators:
            raise ValueError(f"No generator fitted for repeat={repeat}, fold={fold}")
        return self._fold_generators[key].transform(X)


class ManualCVIntegration:
    """
    A complete manual CV training loop that demonstrates how to integrate
    custom feature generation with AutoGluon's model training.

    This approach gives you full control over:
    1. Feature generation CV
    2. Model training CV
    3. OOF prediction generation
    4. Ensemble construction

    Example
    -------
    ```python
    from autogluon.tabular.models import LGBModel

    integrator = ManualCVIntegration(
        feature_generator=MyGPFeatureGenerator(),
        base_model_cls=LGBModel,
        n_splits=5,
    )

    # Train models with synchronized CV
    oof_predictions, models = integrator.fit(X_train, y_train)

    # Make predictions on test set
    test_predictions = integrator.predict(X_test)
    ```
    """

    def __init__(
        self,
        feature_generator: Any,
        base_model_cls: Type[AbstractModel],
        model_hyperparameters: Optional[Dict] = None,
        n_splits: int = 5,
        n_repeats: int = 1,
        stratify: bool = True,
        random_state: int = 0,
    ):
        self.feature_generator = feature_generator
        self.base_model_cls = base_model_cls
        self.model_hyperparameters = model_hyperparameters or {}
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.stratify = stratify
        self.random_state = random_state

        self._fold_models = []
        self._fold_generators = []
        self._full_generator = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **model_fit_kwargs,
    ) -> Tuple[np.ndarray, List[AbstractModel]]:
        """
        Fit models with synchronized CV between feature generation and training.

        Returns
        -------
        oof_pred_proba : np.ndarray
            Out-of-fold predictions.
        models : list
            List of trained fold models.
        """
        n = len(X)
        original_index = X.index.copy()

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        cv_splitter = CVSplitter(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            stratify=self.stratify,
            random_state=self.random_state,
            shuffle=True,
        )
        splits = cv_splitter.split(X, y)

        # Determine output shape
        oof_pred_proba = None
        models = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            repeat = fold_idx // self.n_splits
            fold_in_repeat = fold_idx % self.n_splits
            fold_name = f"S{repeat+1}F{fold_in_repeat+1}"

            logger.info(f"Training fold {fold_name}")

            # 1. Fit feature generator on training fold ONLY
            fold_gen = copy.deepcopy(self.feature_generator)
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_train_transformed = fold_gen.fit_transform(X_train, y_train)

            # 2. Transform validation fold
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            X_val_transformed = fold_gen.transform(X_val)

            # 3. Train model on transformed features
            model = self.base_model_cls(
                name=f"model_{fold_name}",
                hyperparameters=self.model_hyperparameters.copy(),
            )
            model.fit(
                X=X_train_transformed,
                y=y_train,
                X_val=X_val_transformed,
                y_val=y_val,
                **model_fit_kwargs,
            )

            # 4. Generate OOF predictions
            val_pred_proba = model.predict_proba(X_val_transformed)

            # Initialize OOF array if needed
            if oof_pred_proba is None:
                if val_pred_proba.ndim == 1:
                    oof_pred_proba = np.zeros(n, dtype=np.float64)
                else:
                    oof_pred_proba = np.zeros((n, val_pred_proba.shape[1]), dtype=np.float64)

            oof_pred_proba[val_idx] = val_pred_proba

            # Store fitted generator and model
            self._fold_generators.append(fold_gen)
            self._fold_models.append(model)
            models.append(model)

        # Fit full-data generator for inference
        self._full_generator = copy.deepcopy(self.feature_generator)
        self._full_generator.fit_transform(X, y)

        return oof_pred_proba, models

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions by averaging across all fold models.
        """
        if not self._fold_models:
            raise RuntimeError("Must call fit first")

        X_transformed = self._full_generator.transform(X)

        predictions = []
        for model in self._fold_models:
            pred = model.predict_proba(X_transformed)
            predictions.append(pred)

        return np.mean(predictions, axis=0)


# ==============================================================================
# Integration with AutoGluon TabularPredictor
# ==============================================================================

def create_autogluon_compatible_pipeline(
    feature_generator: Any,
    n_splits: int = 5,
    random_state: int = 0,
):
    """
    Create a feature generator that's compatible with AutoGluon's TabularPredictor.

    This function wraps your feature generator in a CV-aware wrapper that
    prevents data leakage when used with AutoGluon.

    Parameters
    ----------
    feature_generator : Any
        Your custom feature generator (with fit_transform/transform API).
    n_splits : int
        Number of CV folds (should match TabularPredictor's num_bag_folds).
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    pipeline : PipelineFeatureGenerator
        AutoGluon-compatible feature generator pipeline.

    Example
    -------
    ```python
    from autogluon.tabular import TabularPredictor

    # Create CV-aware pipeline
    pipeline = create_autogluon_compatible_pipeline(
        feature_generator=MyGPFeatureGenerator(),
        n_splits=5,
    )

    # Use with TabularPredictor
    predictor = TabularPredictor(label='target')
    predictor.fit(
        train_data,
        feature_generator=pipeline,
        num_bag_folds=5,  # Must match n_splits!
    )
    ```
    """
    from autogluon.features.generators import PipelineFeatureGenerator

    # Import our CV-aware wrapper
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from custom_cv_feature_generator import CVAwareFeatureGenerator

    cv_aware_generator = CVAwareFeatureGenerator(
        base_generator=feature_generator,
        n_splits=n_splits,
        stratify=True,
        random_state=random_state,
        keep_original=True,
    )

    pipeline = PipelineFeatureGenerator(
        generators=[cv_aware_generator],
        post_generators=[],
    )

    return pipeline


# ==============================================================================
# Example: Full Integration Workflow
# ==============================================================================

def full_integration_example():
    """
    Complete example showing the recommended integration approach.
    """
    print("=" * 70)
    print("AutoGluon CV-Aware Feature Generator Integration")
    print("=" * 70)

    # Step 1: Define your GP-based feature generator
    print("\nStep 1: Define your custom feature generator")
    print("-" * 50)

    class MyGPFeatureGenerator:
        """
        Example GP-based feature generator.
        Replace with your actual implementation.
        """
        def __init__(self):
            self.feature_programs_ = None
            self.encodings_ = {}

        def fit_transform(self, X, y):
            X = X.copy()

            # Simulate GP feature generation
            # Your actual implementation would use genetic programming here
            for col in X.select_dtypes(include=['number']).columns:
                X[f'{col}_squared'] = X[col] ** 2
                X[f'{col}_log'] = np.log1p(np.abs(X[col]))

            # Target encoding for categoricals (causes data leakage if not handled)
            for col in X.select_dtypes(include=['object', 'category']).columns:
                encoding = y.groupby(X[col]).mean().to_dict()
                self.encodings_[col] = encoding
                X[f'{col}_target_enc'] = X[col].map(encoding).fillna(y.mean())

            return X

        def transform(self, X):
            X = X.copy()

            for col in X.select_dtypes(include=['number']).columns:
                if f'{col}_squared' not in X.columns:
                    X[f'{col}_squared'] = X[col] ** 2
                if f'{col}_log' not in X.columns:
                    X[f'{col}_log'] = np.log1p(np.abs(X[col]))

            for col, encoding in self.encodings_.items():
                if col in X.columns:
                    X[f'{col}_target_enc'] = X[col].map(encoding).fillna(0)

            return X

    print("  Created MyGPFeatureGenerator with target encoding")

    # Step 2: Wrap in CV-aware generator
    print("\nStep 2: Wrap in CV-aware generator")
    print("-" * 50)

    from custom_cv_feature_generator import CVAwareFeatureGenerator

    cv_generator = CVAwareFeatureGenerator(
        base_generator=MyGPFeatureGenerator(),
        n_splits=5,  # Match AutoGluon's num_bag_folds
        stratify=True,
        random_state=42,
        keep_original=True,
    )

    print("  CVAwareFeatureGenerator wraps your generator")
    print("  It will perform internal 5-fold CV to prevent data leakage")

    # Step 3: Create AutoGluon-compatible pipeline
    print("\nStep 3: Create pipeline for AutoGluon")
    print("-" * 50)

    from autogluon.features.generators import PipelineFeatureGenerator

    pipeline = PipelineFeatureGenerator(
        generators=[cv_generator],
        post_generators=[],
    )

    print("  PipelineFeatureGenerator created")

    # Step 4: Usage with TabularPredictor
    print("\nStep 4: Usage with TabularPredictor")
    print("-" * 50)

    usage_code = '''
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(
        label='target',
        problem_type='binary',  # or 'multiclass', 'regression'
    )

    predictor.fit(
        train_data=train_df,
        feature_generator=pipeline,  # Your CV-aware pipeline
        num_bag_folds=5,             # MUST match n_splits in CVAwareFeatureGenerator!
        num_bag_sets=1,
        hyperparameters={
            'GBM': {},
            'CAT': {},
            'XGB': {},
        },
    )

    # Make predictions (uses full-data fitted generator)
    predictions = predictor.predict(test_df)
    '''

    print(usage_code)

    print("\n" + "=" * 70)
    print("Key Points:")
    print("=" * 70)
    print("""
1. CRITICAL: num_bag_folds in predictor.fit() MUST match n_splits in CVAwareFeatureGenerator
   - If they differ, the CV splits won't be synchronized
   - This can lead to subtle data leakage

2. The CVAwareFeatureGenerator:
   - Performs its own internal K-fold CV during fit_transform()
   - Computes OOF encodings for training data (no leakage)
   - Fits on full data for test-time transform()

3. Random states should be aligned for reproducibility:
   - CVAwareFeatureGenerator random_state
   - TabularPredictor random_state (if using AutoGluon's internal CV)

4. For EXACT synchronization with AutoGluon's CV:
   - Use the ManualCVIntegration class
   - Or extend BaggedEnsembleModel to pass fold indices

5. Your GP feature generator must implement:
   - fit_transform(X, y) -> X_transformed
   - transform(X) -> X_transformed
    """)

    return cv_generator, pipeline


if __name__ == "__main__":
    full_integration_example()
