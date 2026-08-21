import logging

import pandas as pd
import pytest

import autogluon.core.utils.utils as utils_mod
from autogluon.core.utils.utils import augment_rare_classes


@pytest.fixture
def utils_log_records():
    """
    Bulletproof log capture: attach a handler directly to the module logger used
    by augment_rare_classes (utils_mod.logger).
    """
    records: list[logging.LogRecord] = []

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = utils_mod.logger

    handler = ListHandler()
    handler.setLevel(logging.DEBUG)

    old_level = logger.level
    old_handlers = list(logger.handlers)

    # Ensure our handler sees everything
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        # (We don't mutate existing handlers; nothing to restore besides our removal.)
        assert logger.handlers == old_handlers


def _messages(records: list[logging.LogRecord]) -> list[str]:
    return [r.getMessage() for r in records]


def test_no_augmentation_returns_same_df_object(utils_log_records):
    X = pd.DataFrame({"y": ["a", "a", "b", "b"], "f": [1, 2, 3, 4]})
    out = augment_rare_classes(X, label="y", threshold=2)

    assert out is X
    assert out.equals(X)

    msgs = _messages(utils_log_records)
    assert any("did not need to duplicate any data" in m for m in msgs)


def test_augment_single_rare_class_adds_expected_rows_and_meets_threshold():
    # a occurs 2, threshold=5 -> add 3
    X = pd.DataFrame({"y": ["a", "a", "b", "b", "b"], "f": [10, 11, 20, 21, 22]})
    out = augment_rare_classes(X, label="y", threshold=5)

    assert len(out) == len(X) + 5
    vc = out["y"].value_counts()
    assert vc["a"] == 5
    assert vc["b"] == 5

    # New rows must be duplicates of existing rows from 'a'
    orig_a = X.loc[X["y"] == "a"].reset_index(drop=True)
    new_rows = out.loc[out.index.difference(X.index)].reset_index(drop=True)
    new_a = new_rows.loc[new_rows["y"] == "a"].reset_index(drop=True)

    for _, row in new_a.iterrows():
        assert ((orig_a == row).all(axis=1)).any()


def test_augment_multiple_rare_classes_total_added_and_threshold_met():
    # b occurs 2, threshold=7 -> add 5
    # c occurs 3, threshold=7 -> add 4
    X = pd.DataFrame(
        {
            "y": ["a"] * 10 + ["b", "b"] + ["c", "c", "c"],
            "f": list(range(10)) + [100, 101] + [200, 201, 202],
        }
    )
    out = augment_rare_classes(X, label="y", threshold=7)

    vc = out["y"].value_counts()
    assert vc["a"] == 10
    assert vc["b"] == 7
    assert vc["c"] == 7
    assert len(out) == len(X) + 9


def test_augmented_indices_unique_and_start_after_max():
    X = pd.DataFrame({"y": ["a", "a", "b"], "f": [1, 2, 3]}, index=[10, 20, 35])
    out = augment_rare_classes(X, label="y", threshold=4)

    new_rows = 5

    assert len(out) == len(X) + new_rows
    assert out.index.is_unique

    start = X.index.max() + 1  # 36
    new_idx = sorted(set(out.index) - set(X.index))
    assert new_idx == [start + i for i in range(new_rows)]


def test_missing_classes_zero_count_are_warned_and_ignored(utils_log_records):
    # Categorical with unused categories -> value_counts includes 0 counts for them.
    y = pd.Categorical(["a", "a", "a"], categories=["a", "b", "c"])
    X = pd.DataFrame({"y": y, "f": [1, 2, 3]})

    # Sanity: ensure our pandas produces 0-count entries
    class_counts = X["y"].value_counts()
    assert (class_counts == 0).any()

    out = augment_rare_classes(X, label="y", threshold=2)

    # Only invalid classes were 0-count; function should return X unchanged.
    assert out is X
    assert out.equals(X)

    msgs = _messages(utils_log_records)
    assert any("0 training examples" in m for m in msgs)
    assert any("These classes will be ignored" in m for m in msgs)
