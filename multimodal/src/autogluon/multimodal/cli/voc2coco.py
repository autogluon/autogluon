"""
Convert VOC format dataset to COCO.
Reference: https://github.com/yukkyo/voc2coco/blob/master/voc2coco.py
With changes:
1. id stored as int by default
2. provide only root_dir, and corresponding simplification
3. split train/val/test
4. Use defusedxml.ElementTree for security concern
5. remove invalid images

To use:
    If you'd like to customize train/val/test ratio. Note test_ratio = 1 - train_ratio - val_ratio.
        python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir> --train_ratio <train_ratio> --val_ratio <val_ratio>
    If you'd like to use the dataset provided train/val/test splits:
        python3 -m autogluon.multimodal.cli.voc2coco --root_dir <root_dir>
"""

import argparse
import json
import os
import random
import re
import subprocess
from typing import Dict, List

import defusedxml.ElementTree as ET
from tqdm import tqdm

from autogluon.multimodal.utils.object_detection import dump_voc_classes, dump_voc_xml_files, process_voc_annotations

DEFAULT_EXT = ".jpg"
MIN_AREA = 4


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
                    rows = f.readlines()
                    if not rows:
                        print(f"Skipping {ann_ids_path}: file is empty")
                    else:
                        ann_ids = []
                        for r in rows:
                            data = r.strip().split()
                            if len(data) == 1:  # Each row is an annotation id
                                ann_ids.append(data[0])
                            elif (
                                len(data) == 2
                            ):  # Each row contains an annotation id and a flag (0 if we do not use this annotation in this split, and 1 if we use it)
                                ann_id, used = data
                                if int(used) == 1:
                                    ann_ids.append(ann_id)
                            else:
                                print(
                                    f"Skipping {ann_ids_path}: file format not recognized. Make sure your annotation follows "
                                    f"VOC format!"
                                )
                                break

                        ann_paths[ann_ids_name] = [os.path.join(ann_dir_path, aid + ".xml") for aid in ann_ids]
        return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext("path")
    if path is None:
        filename = annotation_root.findtext("filename")
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    if not img_name[-4:] in [".jpg", ".png"]:
        img_name = img_name + DEFAULT_EXT
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int("".join(re.findall(r"\d+", img_id)))

    size = annotation_root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    image_info = {"file_name": os.path.join("JPEGImages", img_name), "height": height, "width": width, "id": img_id}
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
        try:
            ann_tree = ET.parse(a_path)
        except:
            ann_tree = ET.parse(os.path.join(root_dir, a_path))
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root, extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info["id"]

        valid_image = False  # remove image without bounding box to speed up mAP calculation
        for obj in ann_root.findall("object"):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            if ann:
                ann.update({"image_id": img_id, "id": bnd_id})
                output_json_dict["annotations"].append(ann)
                bnd_id = bnd_id + 1
                valid_image = True

        if valid_image:
            output_json_dict["images"].append(img_info)

    for label, label_id in label2id.items():
        category_info = {"supercategory": "none", "id": label_id, "name": label}
        output_json_dict["categories"].append(category_info)

    with open(output_jsonpath, "w") as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)
        print(f"The COCO format annotation is saved to {output_jsonpath}")


def main():
    parser = argparse.ArgumentParser(description="This script support converting voc format xmls to coco format json")
    parser.add_argument("--root_dir", type=str, default=None, help="path to VOC format dataset root")
    parser.add_argument("--train_ratio", type=float, default=None, help="training set ratio")
    parser.add_argument("--val_ratio", type=float, default=None, help="validation set ratio")
    parser.add_argument("--ext", type=str, default=".jpg", help="default extension for image file")
    parser.add_argument("--min_area", type=int, default=4, help="min area for a valid bounding box")
    parser.add_argument(
        "--not_extract_num_from_imgid", action="store_true", help="Extract image number from the image filename"
    )
    args = parser.parse_args()

    annpaths_list_path = None
    DEFAULT_EXT = args.ext
    MIN_AREA = args.min_area
    assert DEFAULT_EXT in [".jpg", ".png"]

    if not args.root_dir:
        raise ValueError("Must specify the root of the VOC format dataset.")
    if args.train_ratio is not None:
        assert args.train_ratio >= 0
        assert args.val_ratio >= 0
        assert args.train_ratio + args.val_ratio <= 1
        annpaths_list_path = os.path.join(args.root_dir, "pathlist.txt")
        ## generate pathlist.txt containing all xml file paths
        dump_voc_xml_files(
            voc_annotation_path=os.path.join(args.root_dir, "Annotations"),
            voc_annotation_xml_output_path=annpaths_list_path,
        )

        assert os.path.exists(annpaths_list_path), "FatalError: pathlist.txt does not exist!"

    labels_path = os.path.join(args.root_dir, "labels.txt")
    ## generate labels.txt containing all unique class names
    dump_voc_classes(
        voc_annotation_path=os.path.join(args.root_dir, "Annotations"), voc_class_names_output_path=labels_path
    )

    assert os.path.exists(labels_path), "FatalError: labels.txt does not exist!"

    output_path_fmt = os.path.join(args.root_dir, "Annotations", "%s_cocoformat.json")

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
