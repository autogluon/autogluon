# Custom Cross-Validation Feature

## Overview

This feature adds support for custom cross-validation strategies in AutoGluon, enabling proper temporal validation for time-series data without data leakage.

## Problem

Standard k-fold cross-validation randomly splits data, causing data leakage in temporally-ordered datasets (e.g., financial time series, sales forecasts). Future information leaks into training sets, leading to unrealistically high validation scores and poor production performance.

## Solution

- **CustomCVSplitter**: sklearn-compatible CV splitter accepting user-defined matrices
- **Pre-built strategies**: `forward_chaining_cv()`, `sliding_window_cv()`, `time_series_cv()`
- **TabularPredictor integration**: `custom_cv_matrix` parameter in `.fit()`
- **Full backward compatibility**: All changes are additive with None defaults

## Usage

```python
from autogluon.tabular import TabularPredictor
from autogluon.core.utils.cv import forward_chaining_cv

# Generate forward chaining CV matrix
cv_matrix = forward_chaining_cv(n_samples=len(train_data), n_folds=5, gap=5)

# Train with custom CV
predictor = TabularPredictor(label='target')
predictor.fit(train_data, custom_cv_matrix=cv_matrix)
```

## CV Matrix Format

Each column represents a fold, each row represents a sample:
- `0` = training sample
- `1` = test sample  
- `2` = excluded sample (gap/unused)

## Pre-built Strategies

### Forward Chaining CV

Expanding training window - useful for time series where all historical data is relevant:

```python
from autogluon.core.utils.cv import forward_chaining_cv

cv_matrix = forward_chaining_cv(
    n_samples=1000,
    n_folds=5,
    gap=10  # Gap between train and test to prevent leakage
)
```

### Sliding Window CV

Fixed-size rolling window - useful when only recent history matters:

```python
from autogluon.core.utils.cv import sliding_window_cv

cv_matrix = sliding_window_cv(
    n_samples=1000,
    n_folds=5,
    window_size=200,  # Fixed training window size
    step_size=150,    # Move forward by this amount each fold
    gap=10
)
```

### Time Series CV

Automatic temporal CV from datetime column:

```python
from autogluon.core.utils.cv import time_series_cv
import pandas as pd

cv_matrix = time_series_cv(
    dates=train_data['date'],
    n_folds=5,
    strategy='forward_chaining',
    gap=pd.Timedelta(days=7)
)
```

## Manual Fold Specification

For complete control, create custom CV from indices:

```python
from autogluon.core.utils.cv import create_custom_cv_from_indices

cv_matrix = create_custom_cv_from_indices(
    train_indices_per_fold=[
        [0, 1, 2, 3, 4],      # Fold 1 train
        [0, 1, 2, 3, 4, 5],   # Fold 2 train
    ],
    test_indices_per_fold=[
        [5, 6, 7],            # Fold 1 test
        [8, 9, 10],           # Fold 2 test
    ],
    n_samples=100
)
```

## API Reference

### TabularPredictor.fit()

**New Parameter:**
- `custom_cv_matrix` (pd.DataFrame | np.ndarray | None): Custom CV matrix with shape (n_samples, n_folds). Values: 0=train, 1=test, 2=exclude.

**Notes:**
- Cannot be used with `num_bag_folds` parameter
- Automatically sets `num_bag_folds` from matrix shape
- Matrix must have same number of rows as training data

## Files Changed

### New Files
- `core/src/autogluon/core/utils/cv/` - Custom CV infrastructure
  - `__init__.py` - Package exports
  - `custom_cv_splitter.py` - CustomCVSplitter class
  - `strategies.py` - Pre-built CV strategies
- `core/tests/unittests/utils/cv/` - Unit tests

### Modified Files
- `core/src/autogluon/core/utils/utils.py` - CVSplitter integration
- `core/src/autogluon/core/models/ensemble/bagged_ensemble_model.py` - Bagging support
- `tabular/src/autogluon/tabular/predictor/predictor.py` - Public API

## Testing

Run tests with:
```bash
pytest core/tests/unittests/utils/cv/
pytest tabular/tests/unittests/ -k custom_cv
```

## Backward Compatibility

100% backward compatible. All changes are additive with None defaults. Existing code continues to work unchanged.

## Examples

See test files for more examples:
- `test_cvsplitter_integration.py` - CVSplitter integration
- `test_phase3_predictor_integration.py` - TabularPredictor integration
- `core/tests/unittests/utils/cv/test_custom_cv_integration.py` - Comprehensive tests

