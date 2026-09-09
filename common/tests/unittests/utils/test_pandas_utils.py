from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from autogluon.common.utils import pandas_utils
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage


def test_sample_ratio_ge_1_returns_deep_memory_usage(monkeypatch):
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "yy", "zzz"],  # deep=True matters for object columns
        }
    )

    # Even if get_type_map_raw is broken, this branch should bypass it.
    monkeypatch.setattr(pandas_utils, "get_type_map_raw", lambda _df: {"a": pandas_utils.R_INT, "b": "other"})

    got_1 = get_approximate_df_mem_usage(df, sample_ratio=1)
    got_2 = get_approximate_df_mem_usage(df, sample_ratio=2.0)

    expected = df.memory_usage(deep=True)
    # expected2 = sys.getsizeof(pickle.dumps(df, protocol=4))
    pd.testing.assert_series_equal(got_1, expected)
    pd.testing.assert_series_equal(got_2, expected)


def test_numeric_columns_sampling_returns_shallow_memory_usage(monkeypatch):
    df = pd.DataFrame(
        {
            "i": pd.Series([1, 2, 3, 4], dtype="int64"),
            "f": pd.Series([1.0, 2.0, 3.0, 4.0], dtype="float64"),
        }
    )

    monkeypatch.setattr(
        pandas_utils,
        "get_type_map_raw",
        lambda _df: {"i": pandas_utils.R_INT, "f": pandas_utils.R_FLOAT},
    )

    got = get_approximate_df_mem_usage(df, sample_ratio=0.5)
    expected = df.memory_usage()  # shallow, because no category + no "inexact"

    pd.testing.assert_series_equal(got, expected)


def test_category_column_estimate_matches_formula(monkeypatch):
    df = pd.DataFrame(
        {
            "c": pd.Categorical(["a", "b", "a", "c", "b", "a"]),
        }
    )
    num_rows = len(df)

    monkeypatch.setattr(pandas_utils, "get_type_map_raw", lambda _df: {"c": pandas_utils.R_CATEGORY})

    sample_ratio_in = 0.5
    num_rows_sample = int(np.ceil(sample_ratio_in * num_rows))
    sample_ratio_adj = num_rows_sample / num_rows

    got = get_approximate_df_mem_usage(df, sample_ratio=sample_ratio_in)

    expected = df.memory_usage()  # base shallow usage
    num_categories = max(len(df["c"].cat.categories), 1)
    num_categories_sample = int(np.ceil(sample_ratio_adj * num_categories))
    sample_ratio_cat = num_categories_sample / num_categories

    expected_c = int(
        df["c"].cat.codes.dtype.itemsize * num_rows
        + df["c"].cat.categories[:num_categories_sample].memory_usage(deep=True) / sample_ratio_cat
    )
    expected["c"] = expected_c

    pd.testing.assert_series_equal(got, expected)


def test_inexact_non_object_column_uses_head_deep_scaled(monkeypatch):
    # Non-object "inexact" columns still extrapolate a deep sample. Choose n where
    # ceil(sample_ratio*n) changes the effective ratio: n=6, 0.2 -> ceil(1.2)=2 -> 2/6.
    df = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2020-01-0%d" % i for i in range(1, 7)]),
            "i": pd.Series([1, 2, 3, 4, 5, 6], dtype="int64"),
        }
    )
    num_rows = len(df)
    sample_ratio_in = 0.2
    num_rows_sample = int(np.ceil(sample_ratio_in * num_rows))
    sample_ratio_adj = num_rows_sample / num_rows

    monkeypatch.setattr(
        pandas_utils,
        "get_type_map_raw",
        lambda _df: {"dt": "datetime", "i": pandas_utils.R_INT},
    )

    got = get_approximate_df_mem_usage(df, sample_ratio=sample_ratio_in)

    base = df.memory_usage()  # shallow
    inexact = df[["dt"]].head(num_rows_sample).memory_usage(deep=True)[["dt"]] / sample_ratio_adj
    expected = inexact.combine_first(base)

    pd.testing.assert_series_equal(got, expected)


def test_object_column_counts_each_distinct_object_once(monkeypatch):
    df = pd.DataFrame(
        {
            "obj": ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"],
            "i": pd.Series([1, 2, 3, 4, 5, 6], dtype="int64"),
        }
    )
    num_rows = len(df)
    sample_ratio_in = 0.2
    num_rows_sample = int(np.ceil(sample_ratio_in * num_rows))
    sample_ratio_adj = num_rows_sample / num_rows

    monkeypatch.setattr(
        pandas_utils,
        "get_type_map_raw",
        lambda _df: {"obj": "object", "i": pandas_utils.R_INT},
    )

    got = get_approximate_df_mem_usage(df, sample_ratio=sample_ratio_in)

    sample = df["obj"].to_numpy()[:num_rows_sample]
    unique_bytes = sum(sys.getsizeof(obj) for obj in {id(o): o for o in sample}.values())
    expected = df.memory_usage()  # shallow base
    expected["obj"] = int(sample.itemsize * num_rows + unique_bytes / sample_ratio_adj)

    pd.testing.assert_series_equal(got, expected)


def test_object_column_shared_values_are_not_charged_per_cell():
    """Regression test for #5433.

    ``memory_usage(deep=True)`` charges ``sys.getsizeof(value)`` once per cell, so a column whose
    cells all reference one large string is reported as if each cell held its own copy. pandas'
    C ``read_csv`` parser reuses one object per distinct value, so this is the ordinary case for a
    low-cardinality string column loaded from disk.
    """
    num_rows = 10_000
    shared = "x" * 50_000  # one object, referenced by every cell
    df = pd.DataFrame({"obj": [shared] * num_rows})

    estimate = get_approximate_df_mem_usage(df)["obj"]
    real_upper_bound = df["obj"].to_numpy().itemsize * num_rows + sys.getsizeof(shared) * 8

    assert estimate < real_upper_bound, f"{estimate} not below {real_upper_bound}"
    # The buggy formula charged the 50 KB string to all 10k cells: ~500 MB.
    assert df.memory_usage(deep=True)["obj"] > 100 * estimate


def test_object_column_distinct_values_are_not_undercounted():
    """The shared-object fix must not make high-cardinality columns look free."""
    num_rows = 10_000
    df = pd.DataFrame({"obj": [f"{i}_{'p' * 200}" for i in range(num_rows)]})

    estimate = get_approximate_df_mem_usage(df)["obj"]
    deep = df.memory_usage(deep=True)["obj"]

    # Every value is a distinct object, so the estimate should track deep accounting closely.
    assert 0.9 * deep <= estimate <= 1.1 * deep, f"{estimate} vs {deep}"


def test_memory_usage_shallow_matches_pandas():
    import numpy as np
    import pandas as pd

    from autogluon.common.utils.pandas_utils import _memory_usage_shallow

    df = pd.DataFrame(
        {
            "i8": np.arange(7, dtype=np.int8),
            "i64": np.arange(7),
            "f32": np.arange(7, dtype=np.float32),
            "b": [True, False] * 3 + [True],
            "o": ["x", "yy", None, "zzz", "", "a", "b"],
            "dt": pd.date_range("2020", periods=7),
            "s": pd.Series(["a", "b", None, "d", "e", "f", "g"], dtype="string"),
            "c": pd.Series(list("abababa"), dtype="category"),
            "ni": pd.Series([1, None, 3, 4, 5, 6, 7], dtype="Int64"),
            "sp": pd.arrays.SparseArray([0, 0, 1, 0, 0, 2, 0]),
            "tz": pd.date_range("2020", periods=7, tz="UTC"),
        }
    )
    pd.testing.assert_series_equal(_memory_usage_shallow(df), df.memory_usage())
    sliced = df.iloc[1:5]
    pd.testing.assert_series_equal(_memory_usage_shallow(sliced), sliced.memory_usage())
    empty = df.iloc[:0]
    pd.testing.assert_series_equal(_memory_usage_shallow(empty), empty.memory_usage())


def _constant_columns_reference(df):
    return [column for column in df.columns if len(df[column].unique()) == 1]


def test_get_constant_columns_edge_cases():
    import numpy as np
    import pandas as pd

    from autogluon.common.utils.pandas_utils import get_constant_columns

    df = pd.DataFrame(
        {
            "zero_signs": [0.0, -0.0, 0.0],
            "all_nan": [np.nan, np.nan, np.nan],
            "nan_and_value": [np.nan, 1.0, 1.0],
            "inf": [np.inf, np.inf, np.inf],
            "inf_mixed": [np.inf, -np.inf, np.inf],
            "f32": np.array([1.5, 1.5, 1.5], dtype=np.float32),
            "big_int": [2**53, 2**53 + 1, 2**53],
            "int_const": [7, 7, 7],
            "uint": np.array([3, 3, 3], dtype=np.uint8),
            "bool_const": [True, True, True],
            "bool_var": [True, False, True],
            "obj_none": [None, None, None],
            "obj_none_nan": [None, np.nan, None],
            "obj_const": ["a", "a", "a"],
            "string_na": pd.array(["a", None, "a"], dtype="string"),
            "cat_const": pd.Categorical(["a", "a", "a"]),
            "cat_nan": pd.Categorical(["a", None, "a"]),
            "int64_na": pd.array([pd.NA, pd.NA, pd.NA], dtype="Int64"),
            "nat": pd.Series([pd.NaT, pd.NaT, pd.NaT], dtype="datetime64[ns]"),
            "dt_const": pd.to_datetime(["2020-01-01"] * 3),
            "sparse_const": pd.arrays.SparseArray([0, 0, 0]),
        }
    )
    assert get_constant_columns(df) == _constant_columns_reference(df)
    assert get_constant_columns(df, columns=["int_const", "bool_var", "all_nan"]) == ["int_const", "all_nan"]
    assert get_constant_columns(df.iloc[:0]) == []
    assert get_constant_columns(df.iloc[:1]) == list(df.columns)
    duplicated = pd.concat([df[["int_const", "bool_var"]], df[["int_const"]]], axis=1)
    assert duplicated.columns.has_duplicates
    assert get_constant_columns(duplicated[["bool_var"]]) == []


def test_get_constant_columns_matches_unique_on_random_frames():
    import numpy as np
    import pandas as pd

    from autogluon.common.utils.pandas_utils import get_constant_columns

    def random_column(rng, n):
        kind = rng.integers(10)
        pick = lambda pool: rng.choice(pool, size=n)  # noqa: E731
        if kind == 0:
            return pick(np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 1.5]))
        if kind == 1:
            return pick(np.array([2**53, 2**53 + 1, -1, 0])).astype(np.int64)
        if kind == 2:
            return pick(np.array([True, False]))
        if kind == 3:
            return pd.Series(pick(np.array(["a", "b", None], dtype=object)), dtype=object)
        if kind == 4:
            return pd.Categorical(pick(np.array(["a", "b", None], dtype=object)))
        if kind == 5:
            return pd.array(pick(np.array([1, 2, None], dtype=object)), dtype="Int64")
        if kind == 6:
            return pd.Series(pick(np.array(["2020-01-01", "2020-01-02", None], dtype=object))).astype("datetime64[ns]")
        if kind == 7:
            return pd.array(pick(np.array(["x", "y", None], dtype=object)), dtype="string")
        if kind == 8:
            return pick(np.array([0.25, np.nan], dtype=np.float32))
        return pd.arrays.SparseArray(pick(np.array([0, 0, 3])))

    for seed in range(150):
        rng = np.random.default_rng(seed)
        n = int(rng.choice([1, 2, 3, 20]))
        df = pd.DataFrame({f"c{i}": random_column(rng, n) for i in range(int(rng.integers(1, 9)))})
        assert get_constant_columns(df) == _constant_columns_reference(df), (seed, df.dtypes.to_dict())
