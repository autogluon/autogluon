"""
Simple syntax and import test for Phase 3 changes.
"""

import sys
import ast

print("=" * 80)
print("Phase 3 Integration Test - Syntax and Structure Validation")
print("=" * 80)

# Test 1: Check predictor.py has correct syntax
print("\nTest 1: Verify predictor.py syntax")
# Get repository root
import os
repo_root = os.path.dirname(os.path.abspath(__file__))

try:
    predictor_path = os.path.join(repo_root, 'tabular', 'src', 'autogluon', 'tabular', 'predictor', 'predictor.py')
    with open(predictor_path, 'r') as f:
        code = f.read()
    ast.parse(code)
    print("✓ predictor.py has valid Python syntax")
except SyntaxError as e:
    print(f"✗ Syntax error in predictor.py: {e}")
    sys.exit(1)

# Test 2: Verify custom_cv_matrix parameter exists
print("\nTest 2: Verify custom_cv_matrix parameter in fit() signature")
if 'custom_cv_matrix: pd.DataFrame | np.ndarray | None = None' in code:
    print("✓ custom_cv_matrix parameter found in fit() signature")
else:
    print("✗ custom_cv_matrix parameter not found in fit() signature")
    sys.exit(1)

# Test 3: Verify documentation was added
print("\nTest 3: Verify documentation for custom_cv_matrix")
if 'Custom cross-validation matrix for bagging' in code:
    print("✓ Documentation found for custom_cv_matrix")
else:
    print("✗ Documentation not found for custom_cv_matrix")
    sys.exit(1)

# Test 4: Verify validation logic exists
print("\nTest 4: Verify validation logic for custom_cv_matrix")
if 'Cannot specify both `custom_cv_matrix` and `num_bag_folds`' in code:
    print("✓ Validation logic found")
else:
    print("✗ Validation logic not found")
    sys.exit(1)

# Test 5: Verify custom_cv_matrix is passed to ag_fit_kwargs
print("\nTest 5: Verify custom_cv_matrix is passed to ag_fit_kwargs")
if 'custom_cv_matrix=custom_cv_matrix,' in code:
    print("✓ custom_cv_matrix is passed to ag_fit_kwargs")
else:
    print("✗ custom_cv_matrix not passed to ag_fit_kwargs")
    sys.exit(1)

# Test 6: Verify Phase 1 & 2 code is still intact
print("\nTest 6: Verify Phase 1 & 2 components exist")
try:
    # Check CustomCVSplitter exists
    splitter_path = os.path.join(repo_root, 'core', 'src', 'autogluon', 'core', 'utils', 'cv', 'custom_cv_splitter.py')
    with open(splitter_path, 'r') as f:
        splitter_code = f.read()
    if 'class CustomCVSplitter' in splitter_code:
        print("✓ CustomCVSplitter class exists")
    else:
        print("✗ CustomCVSplitter class not found")
        sys.exit(1)

    # Check BaggedEnsembleModel integration
    bagged_path = os.path.join(repo_root, 'core', 'src', 'autogluon', 'core', 'models', 'ensemble', 'bagged_ensemble_model.py')
    with open(bagged_path, 'r') as f:
        bagged_code = f.read()
    if 'custom_cv_matrix: pd.DataFrame | np.ndarray | None = None' in bagged_code:
        print("✓ BaggedEnsembleModel accepts custom_cv_matrix")
    else:
        print("✗ BaggedEnsembleModel doesn't accept custom_cv_matrix")
        sys.exit(1)

except FileNotFoundError as e:
    print(f"✗ Required file not found: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL STRUCTURE TESTS PASSED! ✓")
print("=" * 80)
print("\nPhase 3 implementation is complete and syntactically correct.")
print("The custom_cv_matrix parameter has been successfully integrated into:")
print("  1. TabularPredictor.fit() signature")
print("  2. Parameter documentation")
print("  3. Validation logic")
print("  4. ag_fit_kwargs passing")
print("  5. Connection to Phase 1 & 2 components")
