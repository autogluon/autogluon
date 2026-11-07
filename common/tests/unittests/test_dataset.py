import io
import sys
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from autogluon.common import TabularDataset
from autogluon.common.loaders import load_pd
from autogluon.common.savers import save_pd


def test_tabular_dataset():
    data = {"col1": [1, 2, 3, 4], "col2": ["a", "b", "b", "c"]}

    df_1 = pd.DataFrame(data)
    df_2 = TabularDataset(data)

    assert isinstance(df_1, pd.DataFrame)
    assert df_1.equals(df_2)
    assert type(df_1) == pd.DataFrame
    assert type(df_1) == type(df_2)


def _make_excel_df() -> pd.DataFrame:
    return pd.DataFrame({"col1": [1, 2, 3, 4], "col2": ["a", "b", "b", "c"]})


def test_tabular_dataset_from_excel(tmp_path):
    df = _make_excel_df()
    path = str(tmp_path / "data.xlsx")
    save_pd.save(path=path, df=df)

    df_loaded = TabularDataset(path)

    assert isinstance(df_loaded, pd.DataFrame)
    pd.testing.assert_frame_equal(df_loaded, df)


def test_load_pd_excel_round_trip(tmp_path):
    df = _make_excel_df()
    path = str(tmp_path / "data.xlsx")
    save_pd.save(path=path, df=df)

    df_loaded = load_pd.load(path)

    pd.testing.assert_frame_equal(df_loaded, df)


def test_load_pd_excel_xlsm(tmp_path):
    df = _make_excel_df()
    path_xlsx = str(tmp_path / "data.xlsx")
    path_xlsm = str(tmp_path / "data.xlsm")
    df.to_excel(path_xlsx, index=False)
    with open(path_xlsx, "rb") as f_in, open(path_xlsm, "wb") as f_out:
        f_out.write(f_in.read())

    df_loaded = load_pd.load(path_xlsm)

    pd.testing.assert_frame_equal(df_loaded, df)


def test_load_pd_excel_explicit_format(tmp_path):
    df = _make_excel_df()
    path = str(tmp_path / "data_no_extension")
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    with open(path, "wb") as f:
        f.write(buffer.getvalue())

    # The file has no recognizable extension, so the format must be explicitly specified
    df_loaded = load_pd.load(path, format="excel")

    pd.testing.assert_frame_equal(df_loaded, df)


def test_load_pd_excel_first_sheet_only(tmp_path):
    df_sheet1 = _make_excel_df()
    df_sheet2 = pd.DataFrame({"other": [10, 20]})
    path = str(tmp_path / "multi_sheet.xlsx")
    with pd.ExcelWriter(path) as writer:
        df_sheet1.to_excel(writer, sheet_name="first", index=False)
        df_sheet2.to_excel(writer, sheet_name="second", index=False)

    df_loaded = load_pd.load(path)

    pd.testing.assert_frame_equal(df_loaded, df_sheet1)


def test_load_pd_excel_kwargs(tmp_path):
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "z", "w", "v"],
        }
    )
    path = str(tmp_path / "data.xlsx")
    df.to_excel(path, index=False)

    # columns_to_keep, usecols, and nrows
    df_loaded = load_pd.load(path, columns_to_keep=["a", "c"], nrows=3, usecols=["a", "b", "c"])
    assert list(df_loaded.columns) == ["a", "c"]
    assert len(df_loaded) == 3
    assert df_loaded["a"].tolist() == [1, 2, 3]

    # skiprows (list form skips data rows but keeps the header)
    df_loaded = load_pd.load(path, skiprows=[1, 2])
    assert list(df_loaded.columns) == ["a", "b", "c"]
    assert df_loaded["a"].tolist() == [3, 4, 5]

    # header and names: the original header row is now data
    df_loaded = load_pd.load(path, header=None, names=["A", "B", "C"], dtype=str)
    assert list(df_loaded.columns) == ["A", "B", "C"]
    assert len(df_loaded) == 6
    assert df_loaded["A"].tolist() == ["a", "1", "2", "3", "4", "5"]

    # dtype
    df_loaded = load_pd.load(path, dtype={"a": str})
    assert df_loaded["a"].tolist() == ["1", "2", "3", "4", "5"]


def test_save_pd_excel_s3():
    df = _make_excel_df()

    fake_object = mock.MagicMock()
    fake_resource = mock.MagicMock()
    fake_resource.Object.return_value = fake_object
    fake_boto3 = SimpleNamespace(resource=mock.MagicMock(return_value=fake_resource))

    with mock.patch.dict(sys.modules, {"boto3": fake_boto3}):
        save_pd.save(path="s3://fake-bucket/fake-prefix/data.xlsx", df=df)

    fake_resource.Object.assert_called_once_with("fake-bucket", "fake-prefix/data.xlsx")
    put_kwargs = fake_object.put.call_args.kwargs
    assert put_kwargs["ACL"] == "bucket-owner-full-control"
    # The uploaded body must be a valid Excel file
    df_loaded = pd.read_excel(io.BytesIO(put_kwargs["Body"]))
    pd.testing.assert_frame_equal(df_loaded, df)


def test_load_pd_excel_s3(tmp_path):
    df = _make_excel_df()
    path = str(tmp_path / "data.xlsx")
    df.to_excel(path, index=False)

    fake_object = mock.MagicMock()

    def fake_download_fileobj(buffer):
        with open(path, "rb") as f:
            buffer.write(f.read())

    fake_object.download_fileobj.side_effect = fake_download_fileobj
    fake_resource = mock.MagicMock()
    fake_resource.Object.return_value = fake_object
    fake_boto3 = SimpleNamespace(resource=mock.MagicMock(return_value=fake_resource))

    with mock.patch.dict(sys.modules, {"boto3": fake_boto3}):
        df_loaded = load_pd.load("s3://fake-bucket/fake-prefix/data.xlsx")

    fake_resource.Object.assert_called_once_with("fake-bucket", "fake-prefix/data.xlsx")
    fake_object.download_fileobj.assert_called_once()
    pd.testing.assert_frame_equal(df_loaded, df)
