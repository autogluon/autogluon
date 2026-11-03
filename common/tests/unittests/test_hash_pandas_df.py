import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl
from autogluon.common.utils import utils
from autogluon.common.utils.utils import hash_pandas_df


def _get_pandas_df(num_rows=20):
    return pd.DataFrame(
        {
            "A": np.arange(num_rows, dtype=np.int64),
            "B": np.random.rand(num_rows),
            "C": np.random.choice(["foo", "bar"], size=num_rows),
        }
    )


def test_when_df_saved_and_loaded_from_disk_then_hash_is_unchanged():
    df = _get_pandas_df()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = str(Path(temp_dir) / "temp_file.pkl")
        save_pkl.save(temp_file, df)
        loaded_df = load_pkl.load(temp_file)
    assert hash_pandas_df(df) == hash_pandas_df(loaded_df)


def test_when_df_copied_then_hash_is_unchanged():
    df = _get_pandas_df()
    df_copied = df.copy()
    assert df is not df_copied
    assert hash_pandas_df(df) == hash_pandas_df(df_copied)


def test_when_df_columns_permuted_then_hash_is_unchanged():
    df = _get_pandas_df()
    df_permuted = df[reversed(df.columns)]
    assert hash_pandas_df(df) == hash_pandas_df(df_permuted)


@pytest.mark.parametrize("df", [None, pd.DataFrame(), _get_pandas_df()])
def test_hash_pandas_df_without_md5(monkeypatch, df):
    def disabled_md5(*args, **kwargs):
        raise ValueError("MD5 is unavailable in FIPS mode")

    monkeypatch.setattr(hashlib, "md5", disabled_md5)
    # Also block the legacy directly imported alias, if present.
    monkeypatch.setattr(utils, "md5", disabled_md5, raising=False)

    digest = hash_pandas_df(df)
    assert len(digest) == 64
    assert int(digest, 16) >= 0
    assert digest == hash_pandas_df(df)


def test_hash_pandas_df_distinguishes_data_and_index():
    df = _get_pandas_df()
    changed_data = df.copy()
    changed_data.loc[0, "A"] += 1
    changed_index = df.copy()
    changed_index.index = changed_index.index + 1

    assert len({hash_pandas_df(df), hash_pandas_df(changed_data), hash_pandas_df(changed_index)}) == 3
    assert hash_pandas_df(None) != hash_pandas_df(pd.DataFrame())
