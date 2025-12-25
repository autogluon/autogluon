"""
CV-Aware Feature Generator Integration with AutoGluon TabularPredictor

This example demonstrates TRUE per-fold feature generation with AutoGluon,
enabling custom feature generators (like GP-based ones with target encoding)
to be used without data leakage during cross-validation.

The key difference from the OOF approach:
- OOF approach: Feature generator does internal CV, then AutoGluon trains on pre-transformed data
- True per-fold: AutoGluon applies the feature generator INSIDE its fold training loop

Usage:
    predictor = TabularPredictor(label='target')
    predictor.fit(
        train_data=train_df,
        cv_feature_generator=YourGPFeatureGenerator(),  # ← New parameter!
        num_bag_folds=5,
        groups='user_id',  # Optional: for group-based CV
    )
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class ExampleGPFeatureGenerator:
    """
    Example feature generator that simulates GP-based feature engineering
    with target encoding.

    Your actual GP-based feature generator should implement:
    - fit_transform(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame
    - transform(X: pd.DataFrame) -> pd.DataFrame

    The generator can create NEW columns (features) during fit_transform/transform.
    """

    def __init__(self):
        self.target_encodings_ = {}
        self.global_mean_ = None
        self.numeric_stats_ = {}
        self._original_columns = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit on training data and transform.
        Called once per fold with ONLY that fold's training data.
        """
        X = X.copy()
        self._original_columns = list(X.columns)
        self.global_mean_ = y.mean()

        # Target encoding for categorical columns (uses label information!)
        for col in X.select_dtypes(include=['object', 'category']).columns:
            # Compute encoding using only the training fold data
            encoding = y.groupby(X[col]).agg(['mean', 'count']).to_dict()
            self.target_encodings_[col] = {
                'mean': encoding['mean'],
                'count': encoding['count'],
            }
            # Apply encoding
            X[f'{col}_te_mean'] = X[col].map(encoding['mean']).fillna(self.global_mean_)
            X[f'{col}_te_count'] = X[col].map(encoding['count']).fillna(0)

        # Polynomial and interaction features for numerics
        numeric_cols = [c for c in X.select_dtypes(include=['number']).columns
                       if not c.endswith(('_te_mean', '_te_count'))]

        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            self.numeric_stats_[col] = {
                'mean': X[col].mean(),
                'std': X[col].std() + 1e-8,
            }
            X[f'{col}_squared'] = X[col] ** 2
            X[f'{col}_normalized'] = (X[col] - self.numeric_stats_[col]['mean']) / self.numeric_stats_[col]['std']

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted encodings.
        Called on validation fold and test data.
        """
        X = X.copy()

        # Apply target encodings
        for col, encoding in self.target_encodings_.items():
            if col in X.columns:
                X[f'{col}_te_mean'] = X[col].map(encoding['mean']).fillna(self.global_mean_)
                X[f'{col}_te_count'] = X[col].map(encoding['count']).fillna(0)

        # Apply numeric transformations
        for col, stats in self.numeric_stats_.items():
            if col in X.columns:
                X[f'{col}_squared'] = X[col] ** 2
                X[f'{col}_normalized'] = (X[col] - stats['mean']) / stats['std']

        return X


def create_sample_data_with_groups():
    """Create sample dataset with categorical features and groups."""
    np.random.seed(42)
    n_samples = 2000
    n_groups = 50

    # Generate classification data
    X_num, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )

    df = pd.DataFrame(X_num, columns=[f'num_{i}' for i in range(5)])

    # Add categorical features
    df['category_A'] = np.random.choice(['cat1', 'cat2', 'cat3', 'cat4'], n_samples)
    df['category_B'] = np.random.choice(['X', 'Y', 'Z'], n_samples)

    # Add group column (for GroupKFold)
    df['user_id'] = np.random.choice(range(n_groups), n_samples)

    # Add target
    df['target'] = y

    return df


def example_usage_with_groups():
    """
    Complete example using cv_feature_generator with groups.

    This demonstrates the new per-fold feature generation capability.
    """
    print("=" * 70)
    print("AutoGluon Per-Fold Feature Generation Example")
    print("=" * 70)

    # Create data
    df = create_sample_data_with_groups()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"\nData created:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Features: {[c for c in train_df.columns if c not in ['target', 'user_id']]}")
    print(f"  Groups column: user_id ({train_df['user_id'].nunique()} unique groups)")

    # Create your feature generator
    gp_generator = ExampleGPFeatureGenerator()

    print("\n" + "-" * 50)
    print("Training with cv_feature_generator")
    print("-" * 50)

    # Use with AutoGluon
    try:
        from autogluon.tabular import TabularPredictor

        predictor = TabularPredictor(
            label='target',
            problem_type='binary',
            eval_metric='accuracy',
        )

        # Key: Pass cv_feature_generator to fit()
        predictor.fit(
            train_data=train_df,
            cv_feature_generator=gp_generator,  # ← Your custom generator
            num_bag_folds=5,                     # Must have bagging enabled
            groups='user_id',                    # Optional: group-based CV
            hyperparameters={
                'GBM': {'num_boost_round': 100},
            },
            time_limit=120,
            verbosity=2,
        )

        print("\n" + "-" * 50)
        print("Evaluation")
        print("-" * 50)

        # Evaluate on test data
        # The predictor will automatically use the fold-specific generators
        test_score = predictor.evaluate(test_df)
        print(f"\nTest accuracy: {test_score['accuracy']:.4f}")

        # Show leaderboard
        print("\nLeaderboard:")
        print(predictor.leaderboard(test_df))

        autogluon_available = True

    except ImportError:
        print("\nAutoGluon not available. Demonstrating the generator manually.")
        autogluon_available = False

    if not autogluon_available:
        # Manual demonstration
        print("\n" + "-" * 50)
        print("Manual demonstration of per-fold feature generation")
        print("-" * 50)

        from sklearn.model_selection import GroupKFold
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score

        X = train_df.drop(['target', 'user_id'], axis=1)
        y = train_df['target']
        groups = train_df['user_id']

        X_test = test_df.drop(['target', 'user_id'], axis=1)
        y_test = test_df['target']

        gkf = GroupKFold(n_splits=5)

        oof_preds = np.zeros(len(X))
        test_preds_sum = np.zeros(len(X_test))

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            print(f"\nFold {fold_idx + 1}/5:")

            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            # Per-fold feature generation (what AutoGluon now does internally)
            import copy
            fold_generator = copy.deepcopy(gp_generator)

            # fit_transform on training fold ONLY
            X_train_transformed = fold_generator.fit_transform(X_train_fold, y_train_fold)

            # transform validation fold
            X_val_transformed = fold_generator.transform(X_val_fold)

            # transform test data
            X_test_transformed = fold_generator.transform(X_test)

            print(f"  Original features: {len(X_train_fold.columns)}")
            print(f"  Transformed features: {len(X_train_transformed.columns)}")

            # Train model
            clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train_transformed, y_train_fold)

            # OOF predictions
            oof_preds[val_idx] = clf.predict(X_val_transformed)

            # Test predictions (averaged across folds)
            test_preds_sum += clf.predict_proba(X_test_transformed)[:, 1]

        # Final test predictions
        test_preds = (test_preds_sum / 5 > 0.5).astype(int)

        print(f"\n\nOOF accuracy: {accuracy_score(y, oof_preds):.4f}")
        print(f"Test accuracy: {accuracy_score(y_test, test_preds):.4f}")

    print("\n" + "=" * 70)
    print("HOW IT WORKS")
    print("=" * 70)
    print("""
With cv_feature_generator, AutoGluon now does TRUE per-fold feature generation:

For each fold in CV:
    1. Split data into train_fold and val_fold
    2. generator = copy.deepcopy(cv_feature_generator)
    3. X_train = generator.fit_transform(X_train_fold, y_train_fold)
    4. X_val = generator.transform(X_val_fold)
    5. model.fit(X_train, y_train_fold)
    6. model.validate(X_val, y_val_fold)
    7. Store generator with model for test-time inference

For test-time prediction:
    For each fold model:
        1. X_test = fold_model._cv_feature_generator.transform(X_test)
        2. predictions = fold_model.predict(X_test)
    3. Return average of all fold predictions

This ensures:
- NO data leakage (target encoding uses only training fold data)
- Full AutoGluon functionality (stacking, HPO, groups, etc.)
- Proper handling of new columns created by generator
""")


if __name__ == "__main__":
    example_usage_with_groups()
