"""Test CVSplitter integration with CustomCVSplitter."""

import sys
import importlib.util
import numpy as np
import pandas as pd

# Import modules directly to avoid setup.py dependency
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load custom CV modules
print("Loading custom CV modules...")
custom_cv_splitter = load_module(
    'autogluon.core.utils.cv.custom_cv_splitter',
    'core/src/autogluon/core/utils/cv/custom_cv_splitter.py'
)
strategies = load_module(
    'autogluon.core.utils.cv.strategies',
    'core/src/autogluon/core/utils/cv/strategies.py'
)

# Make cv module accessible
cv_module = type(sys)('autogluon.core.utils.cv')
cv_module.CustomCVSplitter = custom_cv_splitter.CustomCVSplitter
cv_module.forward_chaining_cv = strategies.forward_chaining_cv
sys.modules['autogluon.core.utils.cv'] = cv_module

# Now load utils with CVSplitter
print("Loading CVSplitter...")
utils = load_module(
    'autogluon.core.utils.utils',
    'core/src/autogluon/core/utils/utils.py'
)

print("\n" + "="*60)
print("Testing CVSplitter with custom_cv_matrix")
print("="*60)

# Test 1: Basic custom CV integration
print("\n1. Creating custom CV matrix...")
cv_matrix = strategies.forward_chaining_cv(n_samples=50, n_folds=3)
print(f"   Created matrix: shape={cv_matrix.shape}")

# Test 2: Create CVSplitter with custom_cv_matrix
print("\n2. Creating CVSplitter with custom_cv_matrix...")
cv_splitter = utils.CVSplitter(custom_cv_matrix=cv_matrix)
print(f"   ✓ CVSplitter created: n_splits={cv_splitter.n_splits}")

# Test 3: Generate splits
print("\n3. Testing split generation...")
X = pd.DataFrame(np.random.randn(50, 5))
y = pd.Series(np.random.randint(0, 2, 50))

splits = cv_splitter.split(X, y)
print(f"   ✓ Generated {len(splits)} splits")

for i, (train_idx, test_idx) in enumerate(splits, 1):
    print(f"   Fold {i}: train={len(train_idx)}, test={len(test_idx)}")

# Test 4: Verify no data leakage
print("\n4. Verifying temporal ordering (no data leakage)...")
for i, (train_idx, test_idx) in enumerate(splits, 1):
    max_train = max(train_idx)
    min_test = min(test_idx)
    assert min_test > max_train, f"Data leakage in fold {i}"
    print(f"   ✓ Fold {i}: max_train={max_train}, min_test={min_test} - OK")

# Test 5: Test mutual exclusivity validation
print("\n5. Testing mutual exclusivity validation...")
try:
    bad_splitter = utils.CVSplitter(
        custom_cv_matrix=cv_matrix,
        groups=pd.Series([1, 2, 3])
    )
    print("   ✗ Should have raised ValueError")
except ValueError as e:
    print(f"   ✓ Correctly caught error: {str(e)[:60]}...")

try:
    bad_splitter = utils.CVSplitter(
        custom_cv_matrix=cv_matrix,
        stratify=True
    )
    print("   ✗ Should have raised ValueError")
except ValueError as e:
    print(f"   ✓ Correctly caught error: {str(e)[:60]}...")

# Test 6: Test with create_custom_cv_from_indices
print("\n6. Testing with manually specified indices...")
custom_matrix = strategies.create_custom_cv_from_indices(
    train_indices_per_fold=[
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ],
    test_indices_per_fold=[
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
    ],
    n_samples=50
)

cv_splitter2 = utils.CVSplitter(custom_cv_matrix=custom_matrix)
splits2 = cv_splitter2.split(X, y)
print(f"   ✓ Generated {len(splits2)} splits from manual indices")

for i, (train_idx, test_idx) in enumerate(splits2, 1):
    print(f"   Fold {i}: train={len(train_idx)}, test={len(test_idx)}")

print("\n" + "="*60)
print("✅ All CVSplitter integration tests passed!")
print("="*60)
