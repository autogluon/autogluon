import pandas as pd

from autogluon.timeseries.models.local.abstract_local_model import hash_ts_dataframe_items

from .test_ts_dataset import DATETIME_INDEX, ITEM_IDS, SAMPLE_TS_DATAFRAME, TARGETS, _build_ts_dataframe


def test_given_dfs_are_identical_when_hashing_then_hash_is_identical():
    df = SAMPLE_TS_DATAFRAME
    df_copy = SAMPLE_TS_DATAFRAME.copy()
    assert df.values.base is not df_copy.values.base
    assert (hash_ts_dataframe_items(df).values == hash_ts_dataframe_items(df_copy).values).all()


def test_given_item_id_is_different_when_hashing_then_hash_is_identical():
    df1 = _build_ts_dataframe(item_ids=ITEM_IDS, datetime_index=DATETIME_INDEX, target=TARGETS)
    df2 = _build_ts_dataframe(item_ids=["A", "B", "C"], datetime_index=DATETIME_INDEX, target=TARGETS)
    assert (hash_ts_dataframe_items(df1).values == hash_ts_dataframe_items(df2).values).all()


def test_given_timestamp_is_different_when_hashing_then_hash_is_different():
    df1 = _build_ts_dataframe(item_ids=ITEM_IDS, datetime_index=DATETIME_INDEX, target=TARGETS)
    df2 = _build_ts_dataframe(
        item_ids=ITEM_IDS,
        datetime_index=tuple(pd.date_range(pd.Timestamp("01-01-2020", freq="D"), periods=3)),
        target=TARGETS,
    )
    assert (hash_ts_dataframe_items(df1).values != hash_ts_dataframe_items(df2).values).all()


def test_given_values_are_different_when_hashing_then_hash_is_different():
    df1 = _build_ts_dataframe(item_ids=ITEM_IDS, datetime_index=DATETIME_INDEX, target=TARGETS)
    df2 = _build_ts_dataframe(item_ids=ITEM_IDS, datetime_index=DATETIME_INDEX, target=TARGETS * 2)
    assert (hash_ts_dataframe_items(df1).values != hash_ts_dataframe_items(df2).values).all()


def test_when_hashing_then_item_id_order_is_preserved():
    df = _build_ts_dataframe(item_ids=["C", "1", "A"], datetime_index=DATETIME_INDEX, target=TARGETS)
    assert hash_ts_dataframe_items(df).index.equals(df.item_ids)


def test_when_items_are_permuted_then_hash_values_are_permuted():
    new_order = [2, 0, 1]
    df1 = _build_ts_dataframe(item_ids=ITEM_IDS, datetime_index=DATETIME_INDEX, target=TARGETS)
    df2 = df1.loc[new_order]
    assert (hash_ts_dataframe_items(df1).loc[new_order].values == hash_ts_dataframe_items(df2).values).all()
