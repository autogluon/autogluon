from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.utils import get_pred_from_proba_df
from autogluon.text import TextPredictor
from io import BytesIO, StringIO
import pandas as pd
import os


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    model = TextPredictor.load(model_dir)
    globals()["column_names"] = model._model.feature_columns

    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):

    if input_content_type == "application/x-parquet":
        buf = BytesIO(request_body)
        data = pd.read_parquet(buf)

    elif input_content_type == "text/csv":
        buf = StringIO(request_body)
        data = pd.read_csv(buf)

    elif input_content_type == "application/json":
        buf = StringIO(request_body)
        data = pd.read_json(buf)

    elif input_content_type == "application/jsonl":
        buf = StringIO(request_body)
        data = pd.read_json(buf, orient='records', lines=True)

    else:
        raise Exception(
            f'{input_content_type} input content type not supported.'
        )
    # no header
    test_columns = sorted(list(data.columns))
    train_columns = sorted(column_names)
    if test_columns != train_columns:
        num_cols = len(data.columns)

        if num_cols != len(column_names):
            raise Exception(
                f"Invalid data format. Input data has {num_cols} while the model expects {len(column_names)}"
            )

        else:
            new_row = pd.DataFrame(data.columns).transpose()
            old_rows = pd.DataFrame(data.values)
            data = pd.concat([new_row, old_rows]).reset_index(drop=True)
            data.columns = column_names

    if model.problem_type == BINARY or model.problem_type == MULTICLASS:
        pred_proba = model.predict_proba(data)
        pred = get_pred_from_proba_df(pred_proba, problem_type=model.problem_type)
        prediction = pd.concat([pred, pred_proba], axis=1)
    else:
        prediction = model.predict(data)
    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()
    if output_content_type == "application/json":
        output = prediction.to_json()
    elif output_content_type == "application/x-parquet":
        if isinstance(prediction, pd.DataFrame):
            prediction.columns = prediction.columns.astype(str)
        output = prediction.to_parquet()
    elif output_content_type == "text/csv":
        output = prediction.to_csv(index=None)
    else:
        raise Exception(f"{output_content_type} content type not supported")

    return output, output_content_type
