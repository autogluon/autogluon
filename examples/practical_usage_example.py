"""
Practical End-to-End Example: Integrating Custom Feature Generator with AutoGluon

This script demonstrates a complete working example of how to integrate
a custom feature generator (like one based on genetic programming) with
AutoGluon's cross-validation pipeline without data leakage.

Run this script to see the integration in action:
    python practical_usage_example.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def create_sample_data():
    """Create sample dataset with categorical and numerical features."""
    np.random.seed(42)
    n_samples = 1000

    # Generate classification data
    X_num, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=2,
        random_state=42
    )

    # Create DataFrame with numerical features
    df = pd.DataFrame(
        X_num,
        columns=[f'num_feat_{i}' for i in range(5)]
    )

    # Add categorical features
    df['cat_feat_1'] = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    df['cat_feat_2'] = np.random.choice(['X', 'Y', 'Z'], n_samples)

    # Add target
    df['target'] = y

    return df


def main():
    print("=" * 70)
    print("Practical Example: CV-Aware Feature Generator with AutoGluon")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Create sample data
    # ------------------------------------------------------------------
    print("\n1. Creating sample dataset...")
    df = create_sample_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    print(f"   Features: {[c for c in train_df.columns if c != 'target']}")

    # ------------------------------------------------------------------
    # Step 2: Define your custom GP-based feature generator
    # ------------------------------------------------------------------
    print("\n2. Defining custom feature generator...")

    class GPFeatureGenerator:
        """
        Example feature generator that uses target encoding.

        In your real implementation, this would be your genetic programming
        based feature generator that creates new features, some of which
        may use target information (like target encoding).

        The key requirement is that this class implements:
        - fit_transform(X, y) -> X_transformed
        - transform(X) -> X_transformed
        """
        def __init__(self):
            self.target_encodings_ = {}
            self.global_mean_ = None
            self.numeric_stats_ = {}

        def fit_transform(self, X, y):
            X = X.copy()
            self.global_mean_ = y.mean()

            # Feature 1: Target encoding for categorical columns
            for col in X.select_dtypes(include=['object', 'category']).columns:
                encoding = y.groupby(X[col]).mean().to_dict()
                self.target_encodings_[col] = encoding
                X[f'{col}_te'] = X[col].map(encoding).fillna(self.global_mean_)

            # Feature 2: Polynomial features for numerics
            numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
            # Remove already created _te columns
            numeric_cols = [c for c in numeric_cols if not c.endswith('_te')]

            for col in numeric_cols[:3]:  # Limit to avoid explosion
                X[f'{col}_sq'] = X[col] ** 2
                # Store stats for normalization in transform
                self.numeric_stats_[col] = {
                    'mean': X[col].mean(),
                    'std': X[col].std()
                }
                X[f'{col}_norm'] = (X[col] - self.numeric_stats_[col]['mean']) / (self.numeric_stats_[col]['std'] + 1e-8)

            return X

        def transform(self, X):
            X = X.copy()

            # Apply target encodings
            for col, encoding in self.target_encodings_.items():
                if col in X.columns:
                    X[f'{col}_te'] = X[col].map(encoding).fillna(self.global_mean_)

            # Apply numeric transformations
            for col, stats in self.numeric_stats_.items():
                if col in X.columns:
                    X[f'{col}_sq'] = X[col] ** 2
                    X[f'{col}_norm'] = (X[col] - stats['mean']) / (stats['std'] + 1e-8)

            return X

    print("   GPFeatureGenerator defined with:")
    print("   - Target encoding for categorical features")
    print("   - Polynomial features for numeric features")

    # ------------------------------------------------------------------
    # Step 3: Wrap with CV-aware generator
    # ------------------------------------------------------------------
    print("\n3. Creating CV-aware wrapper...")

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from custom_cv_feature_generator import CVAwareFeatureGenerator

    cv_generator = CVAwareFeatureGenerator(
        base_generator=GPFeatureGenerator(),
        n_splits=5,      # This MUST match num_bag_folds in predictor.fit()
        stratify=True,
        random_state=42,
        keep_original=True,
    )

    print("   CVAwareFeatureGenerator created with n_splits=5")

    # ------------------------------------------------------------------
    # Step 4: Create AutoGluon pipeline
    # ------------------------------------------------------------------
    print("\n4. Creating AutoGluon feature pipeline...")

    try:
        from autogluon.features.generators import PipelineFeatureGenerator

        pipeline = PipelineFeatureGenerator(
            generators=[cv_generator],
            post_generators=[],
        )

        print("   PipelineFeatureGenerator created successfully")
        autogluon_available = True
    except ImportError:
        print("   AutoGluon not installed, skipping pipeline creation")
        autogluon_available = False

    # ------------------------------------------------------------------
    # Step 5: Train with AutoGluon
    # ------------------------------------------------------------------
    print("\n5. Training with AutoGluon...")

    if autogluon_available:
        try:
            from autogluon.tabular import TabularPredictor

            predictor = TabularPredictor(
                label='target',
                problem_type='binary',
                eval_metric='accuracy',
            )

            predictor.fit(
                train_data=train_df,
                feature_generator=pipeline,
                num_bag_folds=5,         # MUST match n_splits in CVAwareFeatureGenerator!
                num_bag_sets=1,
                num_stack_levels=0,      # Keep simple for demo
                hyperparameters={
                    'GBM': {'num_boost_round': 100},
                },
                time_limit=60,
                verbosity=2,
            )

            print("\n   Training complete!")

            # Evaluate
            test_score = predictor.evaluate(test_df)
            print(f"\n   Test accuracy: {test_score['accuracy']:.4f}")

            # Show leaderboard
            print("\n   Leaderboard:")
            print(predictor.leaderboard(test_df))

        except Exception as e:
            print(f"   Error during training: {e}")
            print("   This is expected if AutoGluon dependencies are not fully installed")

    # ------------------------------------------------------------------
    # Step 6: Manual demonstration (without full AutoGluon)
    # ------------------------------------------------------------------
    print("\n6. Manual demonstration of CV-aware feature generation...")

    # Prepare data
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Demonstrate CV-aware generation
    print("\n   Fitting CV-aware generator...")

    # This is what happens internally:
    # 1. The generator performs internal 5-fold CV
    # 2. For each fold, it fits on training data and transforms validation data
    # 3. This produces OOF features with no data leakage

    X_train_transformed = cv_generator.fit_transform(X_train, y_train)
    X_test_transformed = cv_generator.transform(X_test)

    print(f"\n   Original features: {len(X_train.columns)}")
    print(f"   Transformed features: {len(X_train_transformed.columns)}")
    print(f"\n   New features created:")
    for col in X_train_transformed.columns:
        if col not in X_train.columns:
            print(f"      - {col}")

    # Quick validation with sklearn
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    print("\n   Training sklearn GBM on transformed features...")
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_transformed, y_train)

    train_pred = clf.predict(X_train_transformed)
    test_pred = clf.predict(X_test_transformed)

    print(f"\n   Train accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"   Test accuracy: {accuracy_score(y_test, test_pred):.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key takeaways:

1. Your GP feature generator MUST be wrapped in CVAwareFeatureGenerator
   to prevent data leakage from target encoding features.

2. The n_splits parameter in CVAwareFeatureGenerator MUST match
   num_bag_folds in TabularPredictor.fit().

3. The wrapper performs internal K-fold CV during fit_transform():
   - Fits your generator on each training fold
   - Transforms the validation fold (OOF predictions)
   - Concatenates to get leak-free training features

4. For test data, transform() uses a generator fitted on ALL training data.

5. The pattern works with any feature generator that has:
   - fit_transform(X, y) -> X_transformed
   - transform(X) -> X_transformed
    """)

    return cv_generator


if __name__ == "__main__":
    main()
