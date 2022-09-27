import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import defusedxml.ElementTree as ET
import numpy as np
import pandas as pd

from ..constants import AUTOMM
from .download import download, is_url

logger = logging.getLogger(AUTOMM)


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
    class_names = set()
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
    print(len(img_list))
    d = {"image": [], "rois": [], "image_attr": []}
    for stem in img_list:
        basename = stem + ".xml"
        anno_file = (rpath / "Annotations" / basename).resolve()
        tree = ET.parse(anno_file)
        xml_root = tree.getroot()
        size = xml_root.find("size")
        im_path = xml_root.find("filename").text
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        rois = []
        for obj in xml_root.iter("object"):
            try:
                difficult = int(obj.find("difficult").text)
            except ValueError:
                difficult = 0
            cls_name = obj.find("name").text.strip().lower()
            xml_box = obj.find("bndbox")
            xmin = max(0, float(xml_box.find("xmin").text) - 1) / width
            ymin = max(0, float(xml_box.find("ymin").text) - 1) / height
            xmax = min(width, float(xml_box.find("xmax").text) - 1) / width
            ymax = min(height, float(xml_box.find("ymax").text) - 1) / height
            if xmin >= xmax or ymin >= ymax:
                logger.warning("Invalid bbox: {%s} for {%s}", str(xml_box), anno_file.name)
            else:
                rois.append(
                    {"class": cls_name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "difficult": difficult}
                )
                class_names.update((cls_name,))
        if rois:
            d["image"].append(str(rpath / "JPEGImages" / im_path))
            d["rois"].append(rois)
            d["image_attr"].append({"width": width, "height": height})
    df = pd.DataFrame(d)
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
    coco, entry: dict, min_object_area: Optional[Union[int, float]] = 0, use_crowd: Optional[bool] = False
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
    valid_objs = []
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
            cname = coco.loadCats(obj["category_id"])[0]["name"]
            valid_objs.append(
                {
                    "xmin": xmin / width,
                    "ymin": ymin / height,
                    "xmax": xmax / width,
                    "ymax": ymax / height,
                    "class": cname,
                    "is_crowd": is_crowd,
                }
            )
    return valid_objs


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
    # construct from COCO format
    try_import_pycocotools()
    from pycocotools.coco import COCO

    if isinstance(anno_file, Path):
        anno_file = str(anno_file.expanduser().resolve())
    elif isinstance(anno_file, str):
        anno_file = os.path.expanduser(anno_file)
    coco = COCO(anno_file)

    if isinstance(root, Path):
        root = str(root.expanduser().resolve())
    elif isinstance(root, str):
        root = os.path.abspath(os.path.expanduser(root))
    elif root is None:
        # try to use the default coco structure
        root = os.path.join(os.path.dirname(anno_file), "..")
        logger.info("Using default root folder: %s. Specify `root=...` if you feel it is wrong...", root)
    else:
        raise ValueError("Unable to parse root: {}".format(root))

    # synsets
    classes = [c["name"] for c in coco.loadCats(coco.getCatIds())]
    # load entries
    d = {"image": [], "rois": [], "image_attr": []}
    image_ids = sorted(coco.getImgIds())
    for entry in coco.loadImgs(image_ids):
        if "coco_url" in entry:
            dirname, filename = entry["coco_url"].split("/")[-2:]
            abs_path = os.path.join(root, dirname, filename)
        else:
            abs_path = os.path.join(root, entry["file_name"])
        if not os.path.exists(abs_path):
            raise IOError("Image: {} not exists.".format(abs_path))
        label = _check_load_coco_bbox(coco, entry, min_object_area=min_object_area, use_crowd=use_crowd)
        if not label:
            continue
        d["image_attr"].append({"width": entry["width"], "height": entry["height"]})
        d["image"].append(abs_path)
        d["rois"].append(label)
    df = pd.DataFrame(d)
    return df.sort_values("image").reset_index(drop=True)


def getCOCOCatIDs():
    return [e for e in range(1, 91) if e not in {12, 26, 29, 30, 45, 66, 68, 69, 71, 83}]
