import json
import logging
import os
import warnings
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
    return_class_names: Optional[bool] = False,
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

    class_names = get_voc_classes(root)

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
    df["label"] = df.loc[:, "rois"].copy()

    return df.sort_values("image").reset_index(drop=True)


def get_voc_classes(root):
    if is_url(root):
        root = download(root)
    rpath = Path(root).expanduser()

    labels_file = os.path.join(rpath, "labels.txt")
    if os.path.exists(labels_file):
        with open(labels_file) as f:
            class_names = [line.rstrip().lower() for line in f]
        print(f"using class_names in labels.txt: {class_names}")
    else:
        logger.warning(
            "labels.txt does not exist, using default VOC names. "
            "To create labels.txt, run ls Annotations/* > pathlist.txt in root dir"
        )
        class_names = VOC_CLASSES

    return class_names


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
            # TODO: remove hardcoding here
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
    d = {"image": [], "rois": []}
    image_ids = sorted(coco.getImgIds())
    for entry in coco.loadImgs(image_ids):
        if "coco_url" in entry:
            dirname, filename = entry["coco_url"].split("/")[-2:]
            abs_path = os.path.join(root, dirname, filename)
        else:
            abs_path = os.path.join(root, entry["file_name"])
        if not os.path.exists(abs_path):
            raise IOError("Image: {} not exists.".format(abs_path))
        rois, _ = _check_load_coco_bbox(
            coco,
            entry,
            min_object_area=min_object_area,
            use_crowd=use_crowd,
        )
        if not rois:
            continue
        d["image"].append(abs_path)
        d["rois"].append(rois)
    df = pd.DataFrame(d)
    df["label"] = df.loc[:, "rois"].copy()
    return df.sort_values("image").reset_index(drop=True)


def getCOCOCatIDs(is_voc=False):
    if is_voc:
        return range(20)
    else:
        return [e for e in range(1, 91) if e not in {12, 26, 29, 30, 45, 66, 68, 69, 71, 83}]


COCO_ID_TO_IDX = dict(zip(getCOCOCatIDs(), range(80)))


def COCOId2Idx(id):
    return COCO_ID_TO_IDX[id]


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_NAME_TO_IDX = dict(zip(VOC_CLASSES, range(20)))


def VOCName2Idx(name):
    return VOC_NAME_TO_IDX[name]


def get_image_name_num(path):
    start_idx = path.rfind("/") + 1
    end_idx = path.rindex(".")
    return int(path[start_idx:end_idx])


class COCODataset:
    # refactor data loading into here
    def __init__(self, anno_file):
        self.anno_file = anno_file

        with open(anno_file, "r") as f:
            d = json.load(f)
        image_list = d["images"]
        img_namenum_list = []
        img_id_list = []
        for img in image_list:
            img_namenum_list.append(get_image_name_num(img["file_name"]))
            img_id_list.append(int(img["id"]))
        self.image_namenum_to_id = dict(zip(img_namenum_list, img_id_list))

        self.category_ids = [cat["id"] for cat in d["categories"]]

    def get_image_id_from_path(self, image_path):
        return self.image_namenum_to_id[get_image_name_num(image_path)]

    def save_result(self, ret, data, save_path):
        coco_format_result = []

        for i, row in data.reset_index(drop=True).iterrows():
            image_id = self.get_image_id_from_path(row["image"])
            for j, res in enumerate(ret[i]):
                category_id = self.category_ids[j]
                for bbox in res:
                    coco_format_result.append(
                        {
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": bbox_xyxy_to_xywh(bbox[:4].astype(float).tolist()),
                            "score": float(bbox[4]),
                        }
                    )
        with open(save_path, "w") as f:
            print(f"saving file at {save_path}")
            json.dump(coco_format_result, f)


def cocoeval_torchmetrics(outputs):
    import torch

    from . import MeanAveragePrecision

    map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False)

    preds = []
    target = []
    for img_idx, img_output in enumerate(outputs):  # TODO: refactor here
        img_result = img_output["bbox"]
        boxes = []
        scores = []
        labels = []
        for category_idx, category_result in enumerate(img_result):
            for item_idx, item_result in enumerate(category_result):
                boxes.append(item_result[:4])
                scores.append(float(item_result[4]))
                labels.append(category_idx)
        preds.append(
            dict(
                boxes=torch.tensor(np.array(boxes).astype(float)).float().to("cpu"),
                scores=torch.tensor(scores).float().to("cpu"),
                labels=torch.tensor(labels).long().to("cpu"),
            )
        )

        img_gt = np.array(img_output["label"])
        boxes = img_gt[:, :4]
        labels = img_gt[:, 4]
        target.append(
            dict(
                boxes=torch.tensor(boxes).float().to("cpu"),
                labels=torch.tensor(labels).long().to("cpu"),
            )
        )

    map_metric.update(preds, target)

    return map_metric.compute()


def cocoeval_pycocotools(outputs, data, anno_file, cache_path, metrics):
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

    if isinstance(metrics, list):
        metrics = metrics[0]

    return {metrics: cocoEval.stats[0]}


def cocoeval(outputs, data, anno_file, cache_path, metrics, tool="pycocotools"):
    if (not tool) or tool == "pycocotools":
        return cocoeval_pycocotools(outputs, data, anno_file, cache_path, metrics)
    elif tool == "torchmetrics":
        return cocoeval_torchmetrics(outputs)


def from_coco_or_voc(file_path, splits: Optional[str] = None):
    if os.path.isdir(file_path):
        # VOC use dir as input
        return from_voc(root=file_path, splits=splits)
    else:
        return from_coco(file_path)
