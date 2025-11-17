import pytest

from ....common import get_data_frame_with_item_index, get_prediction_for_df

ITEM_INDEX = ["1", "2", "A", "B"]


@pytest.fixture()
def ensemble_data():
    df = get_data_frame_with_item_index(ITEM_INDEX)  # type: ignore
    preds = get_prediction_for_df(df)

    yield {
        "predictions_per_window": {
            "dummy_model": [preds],
            "dummy_model_2": [preds * 2],
        },
        "data_per_window": [df],
        "model_scores": {"dummy_model": -2.5, "dummy_model_2": -1.0},
    }


@pytest.fixture()
def ensemble_test_data():
    predictions = get_prediction_for_df(
        get_data_frame_with_item_index(ITEM_INDEX)  # type: ignore
    )
    return {"dummy_model": predictions, "dummy_model_2": predictions * 2}
