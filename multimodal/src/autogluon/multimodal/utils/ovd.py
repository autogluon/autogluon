import logging
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def save_ovd_result_df(pred: Iterable, data: Union[pd.DataFrame, Dict], result_path: Optional[str] = None):
    """
    Saving detection results in pd.DataFrame format (per image)

    Parameters
    ----------
    pred
        List containing detection results for one image
    data
        pandas data frame or dict containing the image information to be tested
    result_path
        path to save result
    detection_classes
        all available classes for this detection
    Returns
    -------
    The detection results as pandas DataFrame
    """
    if isinstance(data, dict):
        image_names = data["image"]
    else:
        image_names = data["image"].to_list()
    results = []

    for image_pred, image_name in zip(pred, image_names):
        results.append([image_name, image_pred])
    result_df = pd.DataFrame(results, columns=["image", "bboxes"])

    if result_path:
        result_df.to_csv(result_path, index=False)
        logger.info("Saved detection results to {}".format(result_path))
    return result_df
