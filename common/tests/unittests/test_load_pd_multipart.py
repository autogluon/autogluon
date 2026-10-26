import pandas as pd
import pytest

from autogluon.common.loaders import load_pd


@pytest.fixture
def multipart_data(tmp_path, monkeypatch):
    parts = [pd.DataFrame({"value": [2 * i, 2 * i + 1]}) for i in range(3)]
    paths = [tmp_path / f"part-{i:05d}.parquet" for i in range(3)]
    for part, path in zip(parts, paths):
        part.to_parquet(path, index=False)
    (tmp_path / "metadata.txt").write_text("metadata", encoding="utf-8")
    (tmp_path / "part-directory").mkdir()
    # Pool.map preserves input order; avoid process startup in this file-order regression.
    monkeypatch.setattr(
        load_pd.multiprocessing_utils,
        "execute_multiprocessing",
        lambda workers_count, transformer, chunks, multiprocessing_method: list(map(transformer, chunks)),
    )
    return parts, paths


@pytest.mark.parametrize("reverse_listing", [False, True])
def test_multipart_directory_load_preserves_part_order(tmp_path, monkeypatch, multipart_data, reverse_listing):
    parts, paths = multipart_data
    listing = [path.name for path in paths] + ["metadata.txt", "part-directory"]
    if reverse_listing:
        listing.reverse()
    monkeypatch.setattr(load_pd, "listdir", lambda path: listing)

    result = load_pd.load(str(tmp_path) + "/", worker_count=1)

    pd.testing.assert_frame_equal(result, pd.concat(parts, ignore_index=True))


def test_explicit_multipart_paths_keep_requested_order(multipart_data):
    parts, paths = multipart_data

    result = load_pd.load(paths[::-1], worker_count=1)

    pd.testing.assert_frame_equal(result, pd.concat(parts[::-1], ignore_index=True))


def test_empty_multipart_paths_report_missing_data():
    with pytest.raises(ValueError, match="No data files"):
        load_pd.load([])


@pytest.mark.parametrize("num_parts", [1, 2])
@pytest.mark.parametrize("worker_count", [None, 1, 16])
def test_multipart_load_bounds_workers_by_number_of_parts(monkeypatch, multipart_data, num_parts, worker_count):
    parts, paths = multipart_data
    pool_workers = []
    monkeypatch.setattr(load_pd.multiprocessing, "cpu_count", lambda: 8)

    def execute(workers_count, transformer, chunks, multiprocessing_method):
        pool_workers.append(workers_count)
        return list(map(transformer, chunks))

    monkeypatch.setattr(load_pd.multiprocessing_utils, "execute_multiprocessing", execute)

    result = load_pd.load(paths[:num_parts], worker_count=worker_count)

    pd.testing.assert_frame_equal(result, pd.concat(parts[:num_parts], ignore_index=True))
    expected_workers = min(worker_count or 8, num_parts)
    assert pool_workers == ([] if expected_workers == 1 else [expected_workers])
