import base64
import hashlib
import pandas as pd

from autogluon.core.constants import REGRESSION
from autogluon.core.utils import get_pred_from_proba_df
from autogluon.tabular import TabularPredictor
from io import BytesIO, StringIO
from PIL import Image


def _save_image_and_update_dataframe_column(bytes):
    im_bytes = base64.b85decode(bytes)
    im_hash = hashlib.md5(im_bytes).hexdigest()
    im = Image.open(BytesIO(im_bytes))
    im_name = f'tabular_image_{im_hash}.png'
    im.save(im_name)
    print(f'Image saved as {im_name}')

    return im_name


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    model = TabularPredictor.load(model_dir)
    globals()["column_names"] = model.feature_metadata_in.get_features()

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
        raise ValueError(
            f'{input_content_type} input content type not supported.'
        )
    # TODO: handle no header case when predictor supports retrieving original training columns

    # find image column
    image_column = None
    for column_name, special_types in model.feature_metadata.get_type_map_special().items():
        if 'image_path' in special_types:
            image_column = column_name
            break
    # save image column bytes to disk and update the column with saved path
    if image_column is not None:
        print(f'Detected image column {image_column}')
        data[image_column] = [_save_image_and_update_dataframe_column(bytes) for bytes in data[image_column]]

    if model.problem_type != REGRESSION:
        pred_proba = model.predict_proba(data, as_pandas=True)
        pred = get_pred_from_proba_df(pred_proba, problem_type=model.problem_type)
        pred_proba.columns = [str(c) + '_proba' for c in pred_proba.columns]
        pred.name = str(pred.name) + '_pred' if pred.name is not None else 'pred'
        prediction = pd.concat([pred, pred_proba], axis=1)
    else:
        prediction = model.predict(data, as_pandas=True)
    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()

    if output_content_type == "application/json":
        output = prediction.to_json()
    elif output_content_type == "application/x-parquet":
        prediction.columns = prediction.columns.astype(str)
        output = prediction.to_parquet()
    elif output_content_type == "text/csv":
        output = prediction.to_csv(index=None)
    else:
        raise ValueError(f"{output_content_type} content type not supported")

    return output, output_content_type
