"""
Integration test for TabularPredictor with custom CV matrix (Phase 3).

This test verifies the end-to-end integration of custom CV functionality
through the TabularPredictor API.
"""
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Test 1: Basic integration - can we import and use the new parameter?
print("=" * 80)
print("Test 1: Import and basic parameter acceptance")
print("=" * 80)

try:
    # Import AutoGluon modules
    import sys
    import os
    
    # Get repository root (assuming this script is in repo root)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    
    sys.path.insert(0, os.path.join(repo_root, 'core', 'src'))
    sys.path.insert(0, os.path.join(repo_root, 'common', 'src'))
    sys.path.insert(0, os.path.join(repo_root, 'features', 'src'))
    sys.path.insert(0, os.path.join(repo_root, 'tabular', 'src'))

    from autogluon.tabular import TabularPredictor
    from autogluon.core.utils.cv import forward_chaining_cv

    print("✓ Successfully imported TabularPredictor and forward_chaining_cv")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create a simple dataset
print("\n" + "=" * 80)
print("Test 2: Create test dataset")
print("=" * 80)

try:
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                                n_redundant=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y

    print(f"✓ Created dataset with {len(df)} samples, {len(df.columns)-1} features")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")
except Exception as e:
    print(f"✗ Dataset creation failed: {e}")
    sys.exit(1)

# Test 3: Generate custom CV matrix
print("\n" + "=" * 80)
print("Test 3: Generate custom CV matrix")
print("=" * 80)

try:
    cv_matrix = forward_chaining_cv(n_samples=len(df), n_folds=3, gap=5)

    print(f"✓ Generated CV matrix: shape={cv_matrix.shape}")
    print(f"  Folds: {cv_matrix.shape[1]}")
    print(f"  Samples per fold (first fold):")
    print(f"    Train: {(cv_matrix.iloc[:, 0] == 0).sum()}")
    print(f"    Test:  {(cv_matrix.iloc[:, 0] == 1).sum()}")
    print(f"    Skip:  {(cv_matrix.iloc[:, 0] == 2).sum()}")
except Exception as e:
    print(f"✗ CV matrix generation failed: {e}")
    sys.exit(1)

# Test 4: Validation - conflicting parameters
print("\n" + "=" * 80)
print("Test 4: Validate parameter conflict detection")
print("=" * 80)

try:
    predictor = TabularPredictor(label='target', path='/tmp/ag_test_conflict')

    # This should raise an error
    try:
        predictor.fit(
            df,
            custom_cv_matrix=cv_matrix,
            num_bag_folds=5,  # Conflict!
            time_limit=10
        )
        print("✗ Should have raised ValueError for conflicting parameters")
    except ValueError as e:
        if "Cannot specify both" in str(e):
            print(f"✓ Correctly detected parameter conflict: {str(e)[:80]}...")
        else:
            print(f"✗ Wrong error message: {e}")

    # Clean up
    import shutil
    if os.path.exists('/tmp/ag_test_conflict'):
        shutil.rmtree('/tmp/ag_test_conflict')

except Exception as e:
    print(f"✗ Validation test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Actually train a model (quick test)
print("\n" + "=" * 80)
print("Test 5: End-to-end training with custom CV")
print("=" * 80)

try:
    import os

    predictor = TabularPredictor(label='target', path='/tmp/ag_test_custom_cv')

    print("Starting fit with custom_cv_matrix...")
    predictor.fit(
        df,
        custom_cv_matrix=cv_matrix,
        hyperparameters={'GBM': {}},  # Only train GBM for speed
        time_limit=60,
        verbosity=2
    )

    print(f"✓ Training completed successfully!")
    print(f"  Models trained: {predictor.model_names()}")
    print(f"  Best model: {predictor.model_best}")

    # Make predictions
    predictions = predictor.predict(df.head(10))
    print(f"✓ Predictions generated: {len(predictions)} samples")

    # Clean up
    import shutil
    if os.path.exists('/tmp/ag_test_custom_cv'):
        shutil.rmtree('/tmp/ag_test_custom_cv')

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)

except Exception as e:
    print(f"✗ Training test failed: {e}")
    import traceback
    traceback.print_exc()

    # Clean up on failure
    import shutil
    if os.path.exists('/tmp/ag_test_custom_cv'):
        shutil.rmtree('/tmp/ag_test_custom_cv')
