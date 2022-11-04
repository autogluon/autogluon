"""
Convert VOC format dataset to COCO.
Reference: https://github.com/yukkyo/voc2coco/blob/master/voc2coco.py
With changes:
1. id stored as int by default
2. provide only root_dir, and corresponding simplification
3. split train/val/test
4. Use defusedxml.ElementTree for security concern
5. TODO: remove invalid images?

To use:
1. run in root_dir:
    grep -ERoh '<name>(.*)</name>' ./Annotations | sort | uniq | sed 's/<name>//g' | sed 's/<\/name>//g' > labels.txt
2. run in root_dir:
    ls Annotations/* > pathlist.txt
3. run here:
    python3 voc2coco.py --root_dir <root_dir> --train_ratio <train_ratio> --val_ratio <val_ratio>
"""

import argparse
import defusedxml.ElementTree as ET
import json
import os
import random
import re
import subprocess
from tqdm import tqdm
from typing import Dict, List


MIN_AREA = 4  # TODO: put in arg?


def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, "r") as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str) + 1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(
    root_dir: str,
    annpaths_list_path: str = None,
    train_ratio=0.6,
    val_ratio=0.2,
) -> Dict:
    if annpaths_list_path is not None:
        with open(annpaths_list_path, "r") as f:
            ann_paths = f.read().split()
        random.shuffle(ann_paths)
        N = len(ann_paths)
        num_train = int(N * train_ratio)
        num_val = int(N * val_ratio)
        return {
            "usersplit_train": ann_paths[:num_train],
            "usersplit_val": ann_paths[num_train : num_train + num_val],
            "usersplit_test": ann_paths[num_train + num_val :],
        }
    else:
        ann_ids_folder = os.path.join(root_dir, "ImageSets", "Main")
        ann_dir_path = os.path.join(root_dir, "Annotations")
        ann_paths = {}
        for ann_ids_filename in os.listdir(ann_ids_folder):
            ann_ids_path = os.path.join(ann_ids_folder, ann_ids_filename)
            if os.path.isfile(ann_ids_path) and ann_ids_filename[-4:] == ".txt":
                ann_ids_name = ann_ids_filename[:-4]
                with open(ann_ids_path, "r") as f:
                    ann_ids = f.read().split()
                ann_paths[ann_ids_name] = [os.path.join(ann_dir_path, aid + ".xml") for aid in ann_ids]
        return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext("path")
    if path is None:
        filename = annotation_root.findtext("filename")
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    if filename[-4:] != ".jpg":
        filename = filename + ".jpg"
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r"\d+", img_id)[0])

    size = annotation_root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    image_info = {"file_name": os.path.join("JPEGImages", filename), "height": height, "width": width, "id": img_id}
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext("name")
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find("bndbox")
    xmin = int(float(bndbox.findtext("xmin")))
    ymin = int(float(bndbox.findtext("ymin")))
    xmax = int(float(bndbox.findtext("xmax")))
    ymax = int(float(bndbox.findtext("ymax")))
    if xmin >= xmax or ymin >= ymax:
        return {}
    o_width = xmax - xmin
    o_height = ymax - ymin
    area = o_width * o_height
    if area <= MIN_AREA:
        return {}
    ann = {
        "area": o_width * o_height,
        "iscrowd": 0,
        "bbox": [xmin, ymin, o_width, o_height],
        "category_id": category_id,
        "ignore": 0,
        "segmentation": [],  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(
    annotation_paths: List[str],
    label2id: Dict[str, int],
    output_jsonpath: str,
    extract_num_from_imgid: bool = True,
    root_dir: str = "./",
):
    output_json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print("Start converting !")
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(os.path.join(root_dir, a_path))
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root, extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info["id"]
        output_json_dict["images"].append(img_info)

        for obj in ann_root.findall("object"):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            if ann:
                ann.update({"image_id": img_id, "id": bnd_id})
                output_json_dict["annotations"].append(ann)
                bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {"supercategory": "none", "id": label_id, "name": label}
        output_json_dict["categories"].append(category_info)

    with open(output_jsonpath, "w") as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def main():
    parser = argparse.ArgumentParser(description="This script support converting voc format xmls to coco format json")
    parser.add_argument("--root_dir", type=str, default=None, help="path to VOC format dataset root")
    parser.add_argument("--train_ratio", type=float, default=None, help="training set ratio")
    parser.add_argument("--val_ratio", type=float, default=None, help="validation set ratio")
    parser.add_argument(
        "--not_extract_num_from_imgid", action="store_true", help="Extract image number from the image filename"
    )
    args = parser.parse_args()

    annpaths_list_path = None

    if not args.root_dir:
        raise ValueError("Must specify the root of the VOC format dataset.")
    if args.train_ratio is not None:
        assert args.train_ratio >= 0
        assert args.val_ratio >= 0
        assert args.train_ratio + args.val_ratio <= 1
        annpaths_list_path = os.path.join(args.root_dir, "pathlist.txt")
    labels_path = os.path.join(args.root_dir, "labels.txt")
    output_path_fmt = os.path.join(args.root_dir, "Annotations", "coco_%s.json")

    label2id = get_label2id(labels_path=labels_path)
    ann_paths = get_annpaths(
        root_dir=args.root_dir,
        annpaths_list_path=annpaths_list_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    for mode in ann_paths.keys():
        convert_xmls_to_cocojson(
            annotation_paths=ann_paths[mode],
            label2id=label2id,
            output_jsonpath=output_path_fmt % mode,
            extract_num_from_imgid=(not args.not_extract_num_from_imgid),
            root_dir=args.root_dir,
        )


if __name__ == "__main__":
    main()
