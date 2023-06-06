import numpy as np
import pandas as pd
import tempfile

from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl
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
    with tempfile.TemporaryDirectory() as _:
        tem_pfile = "temp_file.pkl"
        save_pkl.save(tem_pfile, df)
        loaded_df = load_pkl.load(tem_pfile)
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
