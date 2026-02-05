import numpy as np
import pandas as pd

from autogluon.features.generators import AutoMLPipelineFeatureGenerator


def test_isnan_feature_generator_enabled():
    # Setup data with missing values
    # Increase size to avoid heuristic dropping of features
    # Ensure missing patterns are DISTINCT to avoid DropDuplicates removing checking features
    df = pd.DataFrame(
        {
            "int_nan": [1, np.nan, 3, 4] * 5,
            "float_nan": [1.1, 2.2, np.nan, 4.4] * 5,
            "cat_nan": pd.Series(["a", "b", "c", None] * 5, dtype="category"),
            "obj_nan": [None, "y", "z", "w"] * 5,
            "int_clean": [1, 2, 3, 4] * 5,
        }
    )

    # Default should have enable_isnan_features=True
    generator = AutoMLPipelineFeatureGenerator()

    X_out = generator.fit_transform(df)

    # Check that IsNan features are created
    assert "int_nan" in X_out.columns
    assert "__nan__.int_nan" in X_out.columns
    assert "__nan__.float_nan" in X_out.columns
    assert "__nan__.cat_nan" in X_out.columns
    assert "__nan__.obj_nan" in X_out.columns

    # Check that clean columns do NOT generate IsNan feature
    assert "__nan__.int_clean" not in X_out.columns

    # Verify values
    assert X_out["__nan__.int_nan"].iloc[1] == 1
    assert X_out["__nan__.int_nan"].iloc[0] == 0


def test_isnan_feature_generator_disabled():
    df = pd.DataFrame(
        {
            "int_nan": [1, np.nan, 3] * 5,
        }
    )

    # Disable IsNan features
    generator = AutoMLPipelineFeatureGenerator(enable_isnan_features=False)
    X_out = generator.fit_transform(df)

    assert "int_nan" in X_out.columns
    assert "__nan__.int_nan" not in X_out.columns


def test_fit_transform_crash_fix_empty_metadata():
    # Minimal test to ensure no crash with empty features
    # This verifies the fix in AbstractFeatureGenerator.fit_transform
    df_empty = pd.DataFrame()
    generator = AutoMLPipelineFeatureGenerator()
    # Should not crash
    X_out = generator.fit_transform(df_empty)
    assert len(X_out) == 0
