import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import defusedxml.ElementTree as ET
import numpy as np
import pandas as pd
import PIL
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from ..constants import (
    BBOX,
    LABEL,
    MAP,
    MAP_50,
    MAP_75,
    MAP_LARGE,
    MAP_MEDIUM,
    MAP_SMALL,
    MAR_1,
    MAR_10,
    MAR_100,
    MAR_LARGE,
    MAR_MEDIUM,
    MAR_SMALL,
    MEAN_AVERAGE_PRECISION,
)
from .download import download, is_url

logger = logging.getLogger(__name__)


def _get_image_info(image_path: str):
    """
    Get the image width and height info
    Parameters
    ----------
    image_path
        str representing the path to the image
    Returns
    -------
        dict containing image info. None if cannot open image
    """
    info_dict = {}
    try:
        with PIL.Image.open(image_path) as im:
            height, width = im.size
            info_dict["height"] = height
            info_dict["width"] = width
        return info_dict
    except Exception as err:
        warnings.warn(f"Skip image {image_path} due to {err}")
        return None


def get_df_unique_classes(data: pd.DataFrame):
    """
    Get the unique classes in the dataframe for object detection
    Parameters
    ----------
    data
        pd.DataFrame holding the data for object detection
    Returns
    -------
        list of unique classes
    """
    unique_classes = {}
    for idx in range(data.shape[0]):
        row = data.iloc[idx]
        rois = row["rois"]
        for roi in rois:
            _, _, _, _, class_label = roi
            if class_label not in unique_classes:
                unique_classes[class_label] = len(unique_classes)
    return list(unique_classes.keys())


def object_detection_df_to_coco(data: pd.DataFrame, save_path: Optional[str] = None):
    """
    If the user already has dataframe format data and wants to convert to coco format .json files, this function
    completes the task
    Parameters
    ----------
    data
        pd.DataFrame format of object detection data
    save_path
        str path to save the output
    Returns
    -------
        Dict
    """
    output_json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    bbox_count = 0
    unique_classes = {}
    for idx in range(data.shape[0]):
        row = data.iloc[idx]
        image_path = row["image"]
        rois = row["rois"]
        # label = row["label"]
        image_id = idx

        image_info = _get_image_info(image_path)
        if image_info:
            image_entry = {
                "file_name": image_path,
                "height": image_info["height"],
                "width": image_info["width"],
                "id": image_id,
            }
            output_json_dict["images"].append(image_entry)
        else:
            continue

        for roi in rois:
            xmin, ymin, xmax, ymax, class_label = roi
            x, y, w, h = bbox_xyxy_to_xywh([xmin, ymin, xmax, ymax])

            ann = {
                "area": w * h,
                "iscrowd": 0,
                "bbox": [x, y, w, h],
                "category_id": class_label,
                "ignore": 0,
                "segmentation": [],  # This script is not for segmentation
                "image_id": image_id,
                "id": bbox_count,
            }
            bbox_count += 1

            output_json_dict["annotations"].append(ann)

            if class_label not in unique_classes:
                unique_classes[class_label] = len(unique_classes)

    for class_name, id in unique_classes.items():
        output_json_dict["categories"].append({"supercategory": "none", "id": id, "name": class_name})

    if save_path and save_path.endswith(".json"):
        with open(save_path, "w") as fp:
            json.dump(output_json_dict, fp)

    return output_json_dict


def object_detection_data_to_df(data: Union[pd.DataFrame, dict, list, str]) -> pd.DataFrame:
    """
    Construct a dataframe from a data dictionary, json file path (for COCO), folder path (for VOC),
    image path (for single image), list of image paths (for multiple images)
    Parameters
    ----------
    data (dict, str, list)

    Returns
    -------
    a pandas DataFrame with columns "image", "rois", and "label".
    """
    if isinstance(data, dict):
        return from_dict(data)
    if isinstance(data, list):
        return from_list(data)
    if isinstance(data, str):
        if os.path.isdir(data) or data.endswith(".json"):
            return from_coco_or_voc(data)
        return from_str(data)
    if isinstance(data, pd.DataFrame):
        sanity_check_dataframe(data)
        return data

    raise TypeError(
        "Expected data to be an instance of dict, list, str or pd.DataFrame, but got {} of type {}".format(
            data, type(data)
        )
    )


def sanity_check_dataframe(data: pd.DataFrame):
    """
    Checking if the dataframe contains valid headers and values
    Parameters
    ----------
    data
        dataframe holding the data for object detection
    Returns
    -------

    """
    if "image" not in data:
        raise ValueError(f"column 'image' not found in data column names: {data.columns.to_list()}")
    if "rois" not in data and "label" not in data:
        raise ValueError(f"Both column 'rois' and 'label' not found in data column names: {data.columns.to_list()}")
    else:
        if "rois" not in data:
            warnings.warn(
                f"column 'rois' not found in data column names: {data.columns.to_list()}. Copying from 'label' column..."
            )
            data["rois"] = data["label"]
        if "label" not in data:
            warnings.warn(
                f"column 'label' not found in data column names: {data.columns.to_list()}. Copying from 'rois' column..."
            )
            data["label"] = data["rois"]
    assert data.shape[0] > 0, "data frame is empty"


def from_str(data: str) -> pd.DataFrame:
    """
    Construct a dataframe a string representing a single image path
    Parameters
    ----------
    data
        string of the image path
    Returns
    -------
    a pandas DataFrame with columns "image", "rois", and "label".
    """
    d = {"image": [], "rois": [], "label": []}
    d["image"].append(data)
    # Dummy rois
    d["rois"].append([[-1, -1, -1, -1, 0]])
    d["label"].append([[-1, -1, -1, -1, 0]])
    df = pd.DataFrame(d)
    return df.sort_values("image").reset_index(drop=True)


def from_list(data: List[str]) -> pd.DataFrame:
    """
    Construct a dataframe from list of image paths
    Parameters
    ----------
    data
        List containing the image paths
    Returns
    -------
    a pandas DataFrame with columns "image", "rois", and "label".
    """
    d = {"image": [], "rois": [], "label": []}
    for image_name in data:
        d["image"].append(image_name)
        # Dummy rois
        d["rois"].append([[-1, -1, -1, -1, 0]])
        d["label"].append([[-1, -1, -1, -1, 0]])
    df = pd.DataFrame(d)
    return df.sort_values("image").reset_index(drop=True)


def from_dict(data: dict) -> pd.DataFrame:
    """
    Construct a dataframe (dummy) from a data dictionary, with the form {"image": ["img1.jpg", "img2.jpg", ...]}
    Parameters
    ----------
    data
        Dict containing the image paths
    Returns
    -------
    a pandas DataFrame with columns "image", "rois", and "label".
    """
    # TODO: Remove this function after refactoring
    d = {"image": [], "rois": [], "label": []}

    for image in data["image"]:
        d["image"].append(image)
        # Dummy rois
        d["rois"].append([[-1, -1, -1, -1, 0]])
        d["label"].append([[-1, -1, -1, -1, 0]])
    df = pd.DataFrame(d)
    return df.sort_values("image").reset_index(drop=True)


def from_voc(
    root: str,
    splits: Optional[Union[str, tuple]] = None,
    exts: Optional[Union[str, tuple]] = (".jpg", ".jpeg", ".png"),
):
    """
    Construct dataframe from pascal VOC format. Modified from gluon cv.
    Normally you will see a structure like:
    ├── VOC2007
    │   ├── Annotations
    │   ├── ImageSets
    |   |   ├── Main
    |   |   |   ├── train.txt
    |   |   |   ├── test.txt
    │   ├── JPEGImages

    Parameters
    ----------
    root
        The root directory for VOC, e.g., the `VOC2007`. If an url is provided, it will be downloaded and extracted.
    splits
        If given, will search for this name in `ImageSets/Main/`, e.g., ('train', 'test')
    exts
        The supported image formats.

    Returns
    -------
    A dataframe with columns "image", "rois", and "image_attr".
    """
    if is_url(root):
        root = download(root)
    rpath = Path(root).expanduser()
    img_list = []

    class_names = get_detection_classes(root)

    NAME_TO_IDX = dict(zip(class_names, range(len(class_names))))
    name_to_index = lambda name: NAME_TO_IDX[name]

    if splits:
        logger.debug("Use splits: %s for root: %s", str(splits), root)
        if isinstance(splits, str):
            splits = [splits]
        for split in splits:
            split_file = rpath / "ImageSets" / "Main" / split
            if not split_file.resolve().exists():
                split_file = rpath / "ImageSets" / "Main" / (split + ".txt")
            if not split_file.resolve().exists():
                raise FileNotFoundError(split_file)
            with split_file.open(mode="r") as fi:
                img_list += [line.split()[0].strip() for line in fi.readlines()]
    else:
        logger.debug(
            "No split provided, use full image list in %s, with extension %s", str(rpath / "JPEGImages"), str(exts)
        )
        if not isinstance(exts, (list, tuple)):
            exts = [exts]
        for ext in exts:
            img_list.extend([rp.stem for rp in rpath.glob("JPEGImages/*" + ext)])
    d = {"image": [], "rois": []}
    logger.info(f"Number of Images: {len(img_list)}")
    for stem in img_list:
        basename = stem + ".xml"
        anno_file = (rpath / "Annotations" / basename).resolve()
        tree = ET.parse(anno_file)
        xml_root = tree.getroot()
        size = xml_root.find("size")
        im_path = xml_root.find("filename").text
        if "." not in im_path:
            im_path += ".jpg"
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        rois = []
        for obj in xml_root.iter("object"):
            class_label = name_to_index(obj.find("name").text.strip().lower())
            xml_box = obj.find("bndbox")
            xmin = max(0, float(xml_box.find("xmin").text) - 1)
            ymin = max(0, float(xml_box.find("ymin").text) - 1)
            xmax = min(width, float(xml_box.find("xmax").text) - 1)
            ymax = min(height, float(xml_box.find("ymax").text) - 1)
            if xmin >= xmax or ymin >= ymax:
                logger.warning("Invalid bbox: {%s} for {%s}", str(xml_box), anno_file.name)
            else:
                rois.append(
                    [
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                        class_label,
                    ]
                )
        if rois:
            d["image"].append(str(rpath / "JPEGImages" / im_path))
            d["rois"].append(rois)
    df = pd.DataFrame(d)
    df["label"] = df.loc[:, "rois"].copy()  # TODO: remove duplicate column

    return df.sort_values("image").reset_index(drop=True)


def import_try_install(package: str, extern_url: Optional[str] = None):
    """
    Try import the specified package. Modified from gluon cv.
    If the package not installed, try use pip to install and import if success.

    Parameters
    ----------
    package
        The name of the package trying to import.
    extern_url
        The external url if package is not hosted on PyPI.
        For example, you can install a package using:
         "pip install git+http://github.com/user/repo/tarball/master/egginfo=xxx".
        In this case, you can pass the url to the extern_url.

    Returns
    -------
    <class 'Module'>
        The imported python module.
    """
    import tempfile

    import portalocker

    lockfile = os.path.join(tempfile.gettempdir(), package + "_install.lck")
    with portalocker.Lock(lockfile):
        try:
            return __import__(package)
        except ImportError:
            try:
                from pip import main as pipmain
            except ImportError:
                from types import ModuleType

                from pip._internal import main as pipmain

                # fix for pip 19.3
                if isinstance(pipmain, ModuleType):
                    from pip._internal.main import main as pipmain

            # trying to install package
            url = package if extern_url is None else extern_url
            pipmain(["install", "--user", url])  # will raise SystemExit Error if fails

            # trying to load again
            try:
                return __import__(package)
            except ImportError:
                import site
                import sys

                user_site = site.getusersitepackages()
                if user_site not in sys.path:
                    sys.path.append(user_site)
                return __import__(package)
    return __import__(package)


def try_import_pycocotools():
    """
    Tricks to optionally install and import pycocotools. Modified from gluon cv.
    """
    # first we can try import pycocotools
    try:
        import pycocotools as _
    except ImportError:
        # we need to install pycootools, which is a bit tricky
        # pycocotools sdist requires Cython, numpy(already met)
        import_try_install("cython")
        # pypi pycocotools is not compatible with windows
        win_url = "git+https://github.com/zhreshold/cocoapi.git#subdirectory=PythonAPI"
        try:
            if os.name == "nt":
                import_try_install("pycocotools", win_url)
            else:
                import_try_install("pycocotools")
        except ImportError:
            faq = "cocoapi FAQ"
            raise ImportError("Cannot import or install pycocotools, please refer to %s." % faq)


def bbox_xywh_to_xyxy(xywh: Optional[Union[list, tuple, np.ndarray]]):
    """
    Convert bounding boxes from format (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax). Modified from gluon cv.

    Parameters
    ----------
    xywh
        The bbox in format (x, y, w, h).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    A tuple or numpy.ndarray.
    The converted bboxes in format (xmin, ymin, xmax, ymax).
    If input is numpy.ndarray, return is numpy.ndarray correspondingly.
    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError("Bounding boxes must have 4 elements, given {}".format(len(xywh)))
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        return xywh[0], xywh[1], xywh[0] + w, xywh[1] + h
    elif isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError("Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
        xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
        return xyxy
    else:
        raise TypeError("Expect input xywh a list, tuple or numpy.ndarray, given {}".format(type(xywh)))


def bbox_ratio_xywh_to_index_xyxy(
    xywh: Optional[Union[list, tuple, np.ndarray]], image_wh: Optional[Union[list, tuple, np.ndarray]]
):
    """
    Convert bounding boxes from format (x_center_ratio, y_center_ratio, w_ratio, h_ratio) to (xmin, ymin, xmax, ymax) in pixel index.

    Parameters
    ----------
    xywh
        The bbox in format (x_center_ratio, y_center_ratio, w_ratio, h_ratio).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    A tuple or numpy.ndarray.
    The converted bboxes in format (xmin, ymin, xmax, ymax).
    If input is numpy.ndarray, return is numpy.ndarray correspondingly.
    """
    if isinstance(xywh, (tuple, list)):
        assert isinstance(
            image_wh, (tuple, list)
        ), f"image_wh (type: {type(image_wh)} should have the same type with xywh (type: {type(xywh)})"

        if not len(xywh) == 4:
            raise IndexError("Bounding boxes must have 4 elements, given {}".format(len(xywh)))
        if not len(image_wh) == 2:
            raise IndexError("Image Width and Height must have 2 elements, given {}".format(len(image_wh)))

        x, y = xywh[:2]
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        W, H = np.maximum(image_wh[0] - 1, 0), np.maximum(image_wh[1] - 1, 0)

        # ratio to index
        x *= W
        y *= H
        W *= W
        H *= H

        # mid to upper left corner
        x -= W / 2
        y -= H / 2

        return x, y, x + w, y + h  # xywh to xyxy
    elif isinstance(xywh, np.ndarray):
        if isinstance(image_wh, np.ndarray):
            if not xywh.size % 4 == 0:
                raise IndexError("Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
            if not image_wh.size % 2 == 0:
                raise IndexError("Image Width and Height must have n * 2 elements, given {}".format(image_wh.shape))

            xywh = xywh * np.concat([image_wh, image_wh], axis=1)  # ratio to index

            # mid to upper left corner
            xywh[:, :2] -= xywh[:, 2:] / 2

            # xywh to xyxy
            xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:] - 1)))
            return xyxy
        elif isinstance(image_wh, (tuple, list)):
            if not xywh.size % 4 == 0:
                raise IndexError("Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))

            W, H = np.maximum(image_wh[0] - 1, 0), np.maximum(image_wh[1] - 1, 0)

            xywh = xywh * np.array([[W, H, W, H]])  # ratio to index

            # mid to upper left corner
            xywh[:, :2] -= xywh[:, 2:] / 2

            # xywh to xyxy
            xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:] - 1)))
            return xyxy
    else:
        raise TypeError("Expect input xywh a list, tuple or numpy.ndarray, given {}".format(type(xywh)))


def bbox_xyxy_to_xywh(xyxy: Optional[Union[list, tuple, np.ndarray]]):
    """
    Convert bounding boxes from format (xmin, ymin, xmax, ymax) to (x, y, w, h). Modified from gluon cv.

    Parameters
    ----------
    xyxy
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    A tuple or numpy.ndarray.
    The converted bboxes in format (x, y, w, h).
    If input is numpy.ndarray, return is numpy.ndarray correspondingly.
    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError("Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1, y1 = xyxy[0], xyxy[1]
        w, h = xyxy[2] - x1, xyxy[3] - y1
        return x1, y1, w, h
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError("Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        return np.hstack((xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2] + 1))
    else:
        raise TypeError("Expect input xywh a list, tuple or numpy.ndarray, given {}".format(type(xyxy)))


def bbox_clip_xyxy(
    xyxy: Optional[Union[list, tuple, np.ndarray]],
    width: Optional[Union[int, float]],
    height: Optional[Union[int, float]],
):
    """
    Clip bounding box with format (xmin, ymin, xmax, ymax) to specified boundary. Modified from gluon cv.
    All bounding boxes will be clipped to the new region `(0, 0, width, height)`.

    Parameters
    ----------
    xyxy
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.
    width
        Boundary width.
    height
        Boundary height.

    Returns
    -------
    A tuple or numpy.ndarray.
    The clipped bboxes in format (xmin, ymin, xmax, ymax).
    If input is numpy.ndarray, return is numpy.ndarray correspondingly.
    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError("Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[3]))
        return x1, y1, x2, y2
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError("Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[:, 0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[:, 1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[:, 2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[:, 3]))
        return np.hstack((x1, y1, x2, y2))
    else:
        raise TypeError("Expect input xywh a list, tuple or numpy.ndarray, given {}".format(type(xyxy)))


def _check_load_coco_bbox(
    coco,
    entry: dict,
    min_object_area: Optional[Union[int, float]] = 0,
    use_crowd: Optional[bool] = False,
):
    """
    Check and load ground-truth labels. Modified from gluon cv.

    Parameters
    ----------
    coco
        The COCO data class.
    entry
        The image annotation entry.
    min_object_area
        Minimum object area to consider.
    use_crowd
        Use crowd or not.

    Returns
    -------
    Valid objects to consider.
    """
    entry_id = entry["id"]
    # fix pycocotools _isArrayLike which don't work for str in python3
    entry_id = [entry_id] if not isinstance(entry_id, (list, tuple)) else entry_id
    ann_ids = coco.getAnnIds(imgIds=entry_id, iscrowd=None)
    objs = coco.loadAnns(ann_ids)
    # check valid bboxes
    rois = []
    is_crowds = []
    width = entry["width"]
    height = entry["height"]
    for obj in objs:
        if obj["area"] < min_object_area:
            continue
        if obj.get("ignore", 0) == 1:
            continue
        is_crowd = obj.get("iscrowd", 0)
        if not use_crowd and is_crowd:
            continue
        # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
        xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj["bbox"]), width, height)
        # require non-zero box area
        if obj["area"] > 0 and xmax > xmin and ymax > ymin:
            cat_ids = coco.getCatIds()
            id_to_idx = dict(zip(cat_ids, range(len(cat_ids))))
            class_label = id_to_idx[coco.loadCats(obj["category_id"])[0]["id"]]
            rois.append(
                [
                    float(xmin),
                    float(ymin),
                    float(xmax),
                    float(ymax),
                    class_label,
                ]
            )
            is_crowds.append(is_crowd)
    return rois, is_crowds


def from_coco(
    anno_file: Optional[str],
    root: Optional[str] = None,
    min_object_area: Optional[Union[int, float]] = 0,
    use_crowd: Optional[bool] = False,
):
    """
    Load dataset from coco format annotations. Modified from gluon cv.
    The structure of a default coco 2017 dataset looks like:
    .
    ├── annotations
    |   |── instances_val2017.json
    ├── train2017
    └── val2017


    Parameters
    ----------
    anno_file
        The path to the annotation file.
    root
        Root of the COCO folder. The default relative root folder (if set to `None`) is `anno_file/../`.
    min_object_area
        Minimum object area to consider.
    use_crowd
        Use crowd or not.

    Returns
    -------
    A dataframe with columns "image", "rois", and "image_attr".
    """
    # construct coco object from COCO format
    try_import_pycocotools()
    from pycocotools.coco import COCO

    if isinstance(anno_file, Path):
        anno_file = str(anno_file.expanduser().resolve())
    elif isinstance(anno_file, str):
        anno_file = os.path.expanduser(anno_file)
    coco = COCO(anno_file)

    # get data root
    if isinstance(root, Path):
        root = str(root.expanduser().resolve())
    elif isinstance(root, str):
        root = os.path.abspath(os.path.expanduser(root))
    elif root is None:
        # try to use the default coco structure
        root = os.path.join(os.path.dirname(anno_file), "..")
        logger.info(f"Using default root folder: {root}. Specify `root=...` if you feel it is wrong...")
    else:
        raise ValueError("Unable to parse root: {}".format(root))

    # support prediction using data with no annotations
    # note that data with annotation can be used for prediction without any changes
    try:
        num_annotations = len(coco.getAnnIds())
    except KeyError:  # KeyError: 'annotations', there is no annotation entry
        num_annotations = 0

    # load entries
    d = {"image": [], "rois": []}
    image_ids = sorted(coco.getImgIds())
    for entry in coco.loadImgs(image_ids):
        if "coco_url" in entry:
            dirname, filename = entry["coco_url"].split("/")[-2:]
            abs_path = os.path.join(root, dirname, filename)
        else:
            abs_path = os.path.join(root, entry["file_name"])
        if not os.path.exists(abs_path):
            logger.warning(f"File skipped since not exists: {abs_path}.")
            continue
        rois, _ = _check_load_coco_bbox(
            coco,
            entry,
            min_object_area=min_object_area,
            use_crowd=use_crowd,
        )
        if not rois:
            # discard the rows without valid annotation ONLY when data has annotation
            # add default placeholder to data without annotation for prediction
            if not num_annotations:
                d["image"].append(abs_path)
                d["rois"].append([[-1, -1, -1, -1, 0]])  # TODO: maybe remove this placeholder
            continue
        d["image"].append(abs_path)
        d["rois"].append(rois)
    df = pd.DataFrame(d)
    df["label"] = df.loc[:, "rois"].copy()
    return df.sort_values("image").reset_index(drop=True)


def get_image_filename(path: str):
    """
    Get the filename (without extension) from its path.

    Parameters
    ----------
    path
        The path of image.

    Returns
    -------
    The file name of image.
    """
    return Path(path.replace("\\", "/")).stem


class COCODataset:
    # The class that load/save COCO data format.
    # TODO: refactor data loading into here
    def __init__(self, anno_file: str):
        """
        Parameters
        ----------
        anno_file
            The path to COCO format json annotation file.
        """
        self.anno_file = anno_file

        with open(anno_file, "r") as f:
            d = json.load(f)
        image_list = d["images"]
        img_filename_list = []
        img_id_list = []
        for img in image_list:
            img_filename_list.append(get_image_filename(img["file_name"]))
            img_id_list.append(int(img["id"]))
        self.image_filename_to_id = dict(zip(img_filename_list, img_id_list))

        self.category_ids = [cat["id"] for cat in d["categories"]]

    def get_image_id_from_path(self, image_path: str):
        """
        Get image id from its path.

        Parameters
        ----------
        image_path
            Image path.

        Returns
        -------
        Image ID.
        """
        return self.image_filename_to_id[get_image_filename(image_path)]

    def save_result(self, ret: List, data: pd.DataFrame, save_path: str):
        """
        Save COCO format result to given save path.

        Parameters
        ----------
        ret
            The returned prediction result.
        data
            The input data.
        save_path
            The save path given to store COCO format output.
        """
        coco_format_result = []

        for i, row in data.reset_index(drop=True).iterrows():
            image_id = self.get_image_id_from_path(row["image"])

            pred_result = ret[i]
            N_pred = len(pred_result["bboxes"])
            for bbox_idx in range(N_pred):
                coco_format_result.append(
                    {
                        "image_id": image_id,
                        "category_id": self.category_ids[int(pred_result["labels"][bbox_idx].item())],
                        "bbox": bbox_xyxy_to_xywh(pred_result["bboxes"][bbox_idx].tolist()),
                        "score": pred_result["scores"][bbox_idx].item(),
                    }
                )

        with open(save_path, "w") as f:
            print(f"saving file at {save_path}")
            json.dump(coco_format_result, f)


def cocoeval_torchmetrics(outputs: List):
    """
    Evaluate predictor's output using torchmetrics' mAP implementation: https://github.com/Lightning-AI/metrics

    Parameters
    ----------
    outputs
        The predictor's output. It is a list with length equals number of images.

    Returns
    -------
    The mAP result.
    """

    map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False)

    preds = []
    target = []
    for per_img_outputs in outputs:  # TODO: refactor here
        preds.append(
            dict(
                boxes=per_img_outputs[BBOX]["bboxes"].to("cpu"),
                scores=per_img_outputs[BBOX]["scores"].to("cpu"),
                labels=per_img_outputs[BBOX]["labels"].to("cpu"),
            )
        )

        target.append(
            dict(
                boxes=per_img_outputs[LABEL]["bboxes"].to("cpu"),
                labels=per_img_outputs[LABEL]["labels"].to("cpu"),
            )
        )

    map_metric.update(preds, target)

    return map_metric.compute()


def cocoeval_pycocotools(
    outputs: List,
    data: pd.DataFrame,
    anno_file: str,
    cache_path: str,
):
    """
    Evaluate predictor's output using pycocotool's mAP implementation: https://github.com/cocodataset/cocoapi
    Pycocotool's implementation takes COCO format prediction result file as input.
    So here requires a cache_path to store the prediction result file.

    Parameters
    ----------
    outputs
        The predictor's output. It is a list with length equals number of images.
    data
        The input data.
    anno_file
        The path to COCO format json annotation file.
    cache_path
        The cache path to store prediction result in COCO format.

    Returns
    -------
    The mAP result.
    """
    try_import_pycocotools()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    from ..constants import BBOX
    from . import extract_from_output

    coco_dataset = COCODataset(anno_file)

    ret = extract_from_output(ret_type=BBOX, outputs=outputs)

    coco_dataset.save_result(ret, data, cache_path)

    cocoGt = COCO(anno_file)
    cocoDt = cocoGt.loadRes(cache_path)
    annType = "bbox"

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats


def parse_detection_result(
    result: Optional[Union[Dict, np.ndarray]],
):
    if isinstance(result, np.ndarray):
        return {
            MAP: result[0],
            MEAN_AVERAGE_PRECISION: result[0],
            MAP_50: result[1],
            MAP_75: result[2],
            MAP_SMALL: result[3],
            MAP_MEDIUM: result[4],
            MAP_LARGE: result[5],
            MAR_1: result[6],
            MAR_10: result[7],
            MAR_100: result[8],
            MAR_SMALL: result[9],
            MAR_MEDIUM: result[10],
            MAR_LARGE: result[11],
        }
    else:
        result[MEAN_AVERAGE_PRECISION] = result[MAP]
        return result


def cocoeval(
    outputs: List,
    data: pd.DataFrame,
    anno_file: str,
    cache_path: str,
    metrics: Optional[Union[str, List]],
    tool="pycocotools",
):
    """
    Evaluate predictor's output using mAP metrics per COCO's standard.

    Parameters
    ----------
    outputs
        The predictor's output. It is a list with length equals number of images.
    data
        The input data.
    anno_file
        The path to COCO format json annotation file.
    cache_path
        The cache path to store prediction result in COCO format.
    metrics
        The name of metrics to be reported.
    tool
        Use the mAP implementation of "pycocotools" or "torchmetrics".

    Returns
    -------
    The mAP result.
    """
    if (not tool) or tool == "pycocotools":
        result = cocoeval_pycocotools(outputs, data, anno_file, cache_path)
    elif tool == "torchmetrics":
        result = cocoeval_torchmetrics(outputs)
    else:
        raise ValueError(f"Unsupported eval_tool: {tool}")

    result = parse_detection_result(result)

    if metrics:
        if isinstance(metrics, str) and metrics.lower() in result:
            return {metrics.lower(): result[metrics.lower()]}
        elif isinstance(metrics, list):
            return {metric.lower(): result[metric.lower()] for metric in metrics}

    return result


def dump_voc_classes(voc_annotation_path: str, voc_class_names_output_path: str = None) -> [str]:
    """
    Reads annotations for a dataset in VOC format.
    Then
        dumps the unique class names into a labels.txt file.
    Parameters
    ----------
    voc_annotation_path
        root_path for annotations in VOC format
    voc_class_names_output_path
        output path for the labels.txt
    Returns
    -------
    list of strings, [class_name0, class_name1, ...]
    """
    files = os.listdir(voc_annotation_path)
    class_names = set()
    for f in files:
        if f.endswith(".xml"):
            xml_path = os.path.join(voc_annotation_path, f)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for boxes in root.iter("object"):
                class_names.add(boxes.find("name").text)

    sorted_class_names = sorted(list(class_names))
    if voc_class_names_output_path:
        with open(voc_class_names_output_path, "w") as f:
            f.writelines("\n".join(sorted_class_names))

    return sorted_class_names


def dump_voc_xml_files(voc_annotation_path: str, voc_annotation_xml_output_path: str = None) -> [str]:
    """
    Reads annotations for a dataset in VOC format.
    Then
        1. dumps the unique class names into labels.txt file.
        2. dumps the xml annotation file names into pathlist.txt file.
    Parameters
    ----------
    voc_annotation_path
        root_path for annotations in VOC format
    voc_annotation_xml_output_path
        output path for the pathlist.txt
    Returns
    -------
        list of strings, [xml_file0, xml_file1, ...]
    """
    files = os.listdir(voc_annotation_path)
    annotation_path_base_name = os.path.basename(voc_annotation_path)
    xml_file_names = []
    for f in files:
        if f.endswith(".xml"):
            xml_file_names.append(os.path.join(annotation_path_base_name, f))

    if voc_annotation_xml_output_path:
        with open(voc_annotation_xml_output_path, "w") as f:
            f.writelines("\n".join(xml_file_names))

    return xml_file_names


def process_voc_annotations(
    voc_annotation_path: str, voc_class_names_output_path: str, voc_annotation_xml_output_path: str
) -> None:
    """
    Reads annotations for a dataset in VOC format.
    Then
        1. dumps the unique class names into labels.txt file.
        2. dumps the xml annotation file names into pathlist.txt file.
    Parameters
    ----------
    voc_annotation_path
        root_path for annotations in VOC format
    voc_class_names_output_path
        output path for the labels.txt
    voc_annotation_xml_output_path
        output path for the pathlist.txt
    Returns
    -------
        None
    """
    files = os.listdir(voc_annotation_path)
    annotation_path_base_name = os.path.basename(voc_annotation_path)
    class_names = set()
    xml_file_names = []
    for f in files:
        xml_path = os.path.join(voc_annotation_path, f)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for boxes in root.iter("object"):
            class_names.add(boxes.find("name").text)

        xml_file_names.append(os.path.join(annotation_path_base_name, f))

    sorted_class_names = sorted(list(class_names))
    with open(voc_class_names_output_path, "w") as f:
        f.writelines("\n".join(sorted_class_names))

    with open(voc_annotation_xml_output_path, "w") as f:
        f.writelines("\n".join(xml_file_names))


def from_coco_or_voc(file_path: str, splits: Optional[Union[str]] = None):
    """
    Convert the data from coco or voc format to pandas Dataframe.

    Parameters
    ----------
    file_path
        The path to data.
        If it is a file, it should be the COCO format json annotation file.
        If it is a directory, it should be the root folder of VOC format data.
    splits
        The splits to use for VOC format data.

    Returns
    -------
        The data in our pandas Dataframe format.
    """
    if os.path.isdir(file_path):
        # VOC use dir as input
        return from_voc(root=file_path, splits=splits)
    else:
        return from_coco(file_path)


def get_coco_format_classes(sample_data_path: str):
    """
    The all class names for COCO format data.

    Parameters
    ----------
    sample_data_path
        The path to COCO format json annotation file. Could be any split, e.g. train/val/test/....

    Returns
    -------
        All the class names.
    """
    try:
        with open(sample_data_path, "r") as f:
            annotation = json.load(f)
    except:
        raise ValueError(f"Failed to load json from provided json file: {sample_data_path}.")
    return [cat["name"] for cat in annotation["categories"]]


def get_voc_format_classes(root: str):
    """
    The all class names for VOC format data.

    Parameters
    ----------
    root
        The path to the root directory of VOC data.

    Returns
    -------
        All the class names.
    """
    if is_url(root):
        root = download(root)
    rpath = Path(root).expanduser()

    labels_file = os.path.join(rpath, "labels.txt")
    if os.path.exists(labels_file):
        with open(labels_file) as f:
            class_names = [line.rstrip().lower() for line in f]
        print(f"using class_names in labels.txt: {class_names}")
    else:
        ## read the class names and save results
        logger.warning(
            "labels.txt does not exist, using default VOC names. "
            "Creating labels.txt by scanning the directory: {}".format(os.path.join(root, "Annotations"))
        )
        class_names = dump_voc_classes(
            voc_annotation_path=os.path.join(root, "Annotations"), voc_class_names_output_path=labels_file
        )

    return class_names


def get_detection_classes(sample_data_path):
    """
    The all class names for given data.

    Parameters
    ----------
    sample_data_path
        If it is a file, it is the path to COCO format json annotation file. Could be any split, e.g. train/val/test/....
        If it is a directory, it is path to the root directory of VOC data.

    Returns
    -------
        All the class names.
    """
    if isinstance(sample_data_path, str):
        if os.path.isdir(sample_data_path):
            return get_voc_format_classes(sample_data_path)
        else:
            return get_coco_format_classes(sample_data_path)
    elif isinstance(sample_data_path, pd.DataFrame):
        return get_df_unique_classes(sample_data_path)


def visualize_detection(
    pred: pd.DataFrame,
    detection_classes: List[str],
    conf_threshold: float,
    visualization_result_dir: str,
):
    """
    Visualize detection results for one image, and save to visualization_result_dir

    Parameters
    ----------
    pred
        Detection results as in pd.DataFrame format
    detection_classes
        All classes for detection
    conf_threshold
        Bounding box confidence threshold to filter unwanted detections
    visualization_result_dir
        Directory to save the visualization results
    Returns
    -------
    an List of np.ndarray of visualized images
    """
    try:
        import cv2
    except:
        raise ImportError("No module named: cv2. Please install cv2 by 'pip install opencv-python'")

    if not os.path.exists(visualization_result_dir):
        os.makedirs(visualization_result_dir, exist_ok=True)

    classname2idx = {classname: i for i, classname in enumerate(detection_classes)}
    idx2classname = {i: classname for i, classname in enumerate(detection_classes)}

    visualized_images = []
    for i in range(len(pred)):
        image_path = pred.iloc[i]["image"]
        image_pred = pred.iloc[i]["bboxes"]
        im = cv2.imread(image_path)
        tlwhs = []
        obj_ids = []
        conf_scores = []
        for data in image_pred:
            if data["score"] > conf_threshold:
                obj_ids.append(classname2idx[data["class"]])
                tlwhs.append(bbox_xyxy_to_xywh(data["bbox"]))
                conf_scores.append(data["score"])
        visualized_im = plot_detections(im, tlwhs, obj_ids, idx2classname, conf_threshold, scores=conf_scores)
        visualized_images.append(visualized_im)
        imgname = os.path.basename(image_path)
        cv2.imwrite(os.path.join(visualization_result_dir, imgname), visualized_im)
    logger.info("Saved visualizations to {}".format(visualization_result_dir))
    return visualized_images


def plot_detections(
    image,
    tlwhs,
    obj_ids,
    idx2classname,
    conf_threshold,
    scores=None,
    text_scale=0.75,
    text_thickness=1,
    line_thickness=2,
    alpha=0.5,
):
    """
    Plot the detections on to the corresponding image

    Parameters
    ----------
    image
        np.ndarray: np array containing the image data
    tlwhs
        list: list containing the bounding boxes in (x1, y1, x2, y2) format
    obj_ids
        list: list containing the class indices of the bounding boxes, length should match tlwhs
    idx2classname
        dict: maps obj_ids to class name (str)
    conf_threshold
        float: confidence threshold to filter bounding boxes
    scores
        list: confidence scores of the bounding boxes, length should match tlwhs
    text_scale
        float: font size of the text display
    text_thickness
        int: font weight of the text display
    line_thickness
        int: line width of the bounding box display
    alpha
        float: opacity of the text display background color

    Returns
    -------
    an np.ndarray of visualized image
    """
    # TODO: Convert to use mmdet package
    try:
        import cv2
    except:
        raise ImportError("No module named: cv2. Please install cv2 by 'pip install opencv-python'")
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    font = cv2.FONT_HERSHEY_DUPLEX
    text_scale = text_scale if im_w > 500 else text_scale * 0.8

    title = "num_det: %d conf: %.2f" % (len(tlwhs), conf_threshold)
    im = add_text_with_bg_color(
        im=im,
        text=title,
        tl=(0, 0),
        bg_color=(0, 0, 0),
        alpha=alpha,
        font=font,
        text_scale=text_scale,
        text_thickness=text_thickness,
    )

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = idx2classname[obj_ids[i]]
        if scores is not None:
            id_text = id_text + ",{:.3f}".format(float(scores[i]))
        color = get_color(abs(obj_id))
        im = add_bbox_with_alpha(
            im=im, tl=intbox[0:2], br=intbox[2:4], line_color=color, alpha=alpha, line_thickness=line_thickness
        )
        im = add_text_with_bg_color(
            im=im,
            text=id_text,
            tl=(intbox[0], intbox[1]),
            bg_color=color,
            alpha=0.75,
            font=font,
            text_scale=text_scale,
            text_thickness=text_thickness,
        )
    return im


def add_bbox_with_alpha(im: np.ndarray, tl: tuple, br: tuple, line_color: tuple, alpha: float, line_thickness: int):
    """
    draw one box borders with transparency (alpha)

    Parameters
    ----------
    im
        np.ndarray: the image to draw bbox on
    tl
        tuple: bottom right corner of the bounding box: tl = (x1, y1)
    br
        tuple: bottom right corner of the bounding box: br = (x1, y1)
    line_color
        tuple: the color of the box borders, e.g. (0, 0, 0)
    alpha
        float: the opacity of the bbox borders
    line_thickness:
        int: thickness of the border
    Returns
    -------
    an np.ndarray of image with added bbox
    """
    try:
        import cv2
    except:
        raise ImportError("No module named: cv2. Please install cv2 by 'pip install opencv-python'")
    overlay = im.copy()
    cv2.rectangle(overlay, tl, br, line_color, thickness=line_thickness)
    im = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)
    return im


def add_text_with_bg_color(
    im: np.ndarray,
    text: str,
    tl: tuple,
    bg_color: tuple,
    alpha: float,
    font,
    text_scale: float,
    text_thickness: int,
    text_vert_padding: int = None,
):
    """
    Add text to im with background color

    Parameters
    ----------
    im
        np.ndarray: the image to add text on
    text
        string: the text content
    tl
        tuple: top left corner of the text region, tl = (x1, y1)
    bg_color
        tuple: the color of the background, e.g. (0, 0, 0)
    alpha
        float: the opacity of the background
    font
        the font of the text, e.g. cv2.FONT_HERSHEY_DUPLEX
    text_scale
        float: the scale (font size) of the text, e.g. 0.75
    text_thickness
        int: the font weight of the text, e.g. 1
    text_vert_padding
        int: vertical padding of the text on each side
    Returns
    -------
    an np.ndarray of image with added text
    """
    try:
        import cv2
    except:
        raise ImportError("No module named: cv2. Please install cv2 by 'pip install opencv-python'")

    x1, y1 = tl

    overlay = im.copy()
    text_size, _ = cv2.getTextSize(text, font, float(text_scale), text_thickness)
    text_w, text_h = text_size

    text_vert_padding = text_vert_padding if text_vert_padding else int(text_h * 0.1)

    y1 = max(y1 - text_h - text_vert_padding * 2, 0)

    cv2.rectangle(overlay, (x1, y1), (x1 + text_w, y1 + text_h + text_vert_padding * 2), bg_color, -1)
    im = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)
    cv2.putText(
        im, text, (x1, y1 + text_h + text_vert_padding), font, text_scale, (255, 255, 255), thickness=text_thickness
    )
    return im


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def save_result_df(
    pred: Iterable, data: Union[pd.DataFrame, Dict], detection_classes: List[str], result_path: Optional[str] = None
):
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
    idx2classname = {i: classname for (i, classname) in enumerate(detection_classes)}

    for image_pred, image_name in zip(pred, image_names):
        box_info = []
        N_preds = len(image_pred["bboxes"])
        for i in range(N_preds):
            box_info.append(
                {
                    "class": idx2classname[image_pred["labels"][i].item()],
                    "class_id": image_pred["labels"][i].item(),
                    "bbox": image_pred["bboxes"][i].tolist(),
                    "score": image_pred["scores"][i].item(),
                }
            )
        results.append([image_name, box_info])
    result_df = pd.DataFrame(results, columns=["image", "bboxes"])
    if result_path:
        result_df.to_csv(result_path, index=False)
        logger.info("Saved detection results to {}".format(result_path))
    return result_df


def save_result_coco_format(detection_data_path, pred, result_path):
    coco_dataset = COCODataset(detection_data_path)
    result_name, _ = os.path.splitext(result_path)
    result_path = result_name + ".json"
    coco_dataset.save_result(pred, from_coco_or_voc(detection_data_path, "test"), save_path=result_path)
    logger.info(25, f"Saved detection result to {result_path}")


def save_result_voc_format(pred, result_path):
    result_name, _ = os.path.splitext(result_path)
    result_path = result_name + ".npy"
    np.save(result_path, pred)
    logger.info(25, f"Saved detection result to {result_path}")


def convert_pred_to_xywh(pred: Optional[List]):
    if not pred:
        return pred
    for i, pred_per_image in enumerate(pred):
        pred[i]["bboxes"] = bbox_xyxy_to_xywh(pred_per_image["bboxes"].detach().numpy())
    return pred
