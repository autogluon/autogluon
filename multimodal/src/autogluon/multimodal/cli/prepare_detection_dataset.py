import argparse
import os
import shutil

import requests


def get_root_dir(output_dir=None, new_folder_name=None):
    if not output_dir:
        root_dir = "./"
    elif os.path.isdir(output_dir):
        if new_folder_name:
            root_dir = os.path.join(output_dir, new_folder_name)
            if not os.path.exists(root_dir):
                os.mkdir(root_dir)
        else:
            root_dir = output_dir
    else:
        raise ValueError(f"{output_dir} is not a valid directory")
    return root_dir


def get_fname_from_path_or_url(path_or_url):
    fname_start = path_or_url.rfind("/") + 1
    fname = path_or_url[fname_start:]
    return fname


def download_one_url(url, root_dir, fname=None):
    if not fname:
        fname = get_fname_from_path_or_url(url)
    output_path = os.path.join(root_dir, fname)
    print(f"Downloading {fname}...")

    r = requests.get(url, timeout=(10, 1000))
    with open(output_path, "wb") as f:
        f.write(r.content)

    return output_path


def download_urls(urls, root_dir, fnames=[]):
    output_paths = []
    if isinstance(urls, str):
        urls = [urls]
    for i, url in enumerate(urls):
        output_path = download_one_url(url, root_dir, fname=fnames[i] if fnames else None)
        output_paths.append(output_path)
    return output_paths


def unpack(archived_file_paths, root_dir):
    for archived_file_path in archived_file_paths:
        fname = get_fname_from_path_or_url(archived_file_path)
        print(f"extracting {fname}...")
        shutil.unpack_archive(archived_file_path, root_dir)


def remove_archived_file_paths(archived_file_paths):
    for archived_file_path in archived_file_paths:
        fname = get_fname_from_path_or_url(archived_file_path)
        print(f"removing {fname}...")
        os.remove(archived_file_path)


def prepare_dataset(output_dir, new_folder_name, urls, fnames=[]):
    root_dir = get_root_dir(output_dir=output_dir, new_folder_name=new_folder_name)
    archived_file_paths = download_urls(urls, root_dir, fnames=fnames)
    unpack(archived_file_paths, root_dir)
    remove_archived_file_paths(archived_file_paths)


def prepare_coco17(output_dir):
    print("Preparing COCO17 dataset...")
    urls = [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/zips/test2017.zip",
        "http://images.cocodataset.org/zips/unlabeled2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
        "http://images.cocodataset.org/annotations/image_info_test2017.zip",
        "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip",
    ]
    prepare_dataset(output_dir=output_dir, new_folder_name="coco17", urls=urls)


def prepare_voc07(output_dir):
    print("Preparing VOC07 dataset...")
    urls = [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    ]
    prepare_dataset(output_dir=output_dir, new_folder_name=None, urls=urls)


def prepare_voc12(output_dir):
    print("Preparing VOC12 dataset...")
    urls = [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    ]
    prepare_dataset(output_dir=output_dir, new_folder_name=None, urls=urls)


def prepare_voc0712(output_dir):
    prepare_voc07(output_dir)
    prepare_voc12(output_dir)


def prepare_watercolor(output_dir):
    print("Preparing Watercolor dataset...")
    urls = [
        "http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/cross_domain_detection/datasets/watercolor.zip",
    ]
    prepare_dataset(output_dir=output_dir, new_folder_name=None, urls=urls)


def prepare_pothole(output_dir):
    print("Preparing Pothole dataset...")
    # urls = ["https://www.kaggle.com/datasets/andrewmvd/pothole-detection/download?datasetVersionNumber=1"]
    urls = ["https://automl-mm-bench.s3.amazonaws.com/object_detection/dataset/pothole.zip"]
    prepare_dataset(output_dir=output_dir, new_folder_name=None, urls=urls, fnames=[])


def main(dataset_name, output_dir):
    if dataset_name.lower() in ["coco", "coco17", "coco2017"]:
        prepare_coco17(output_dir=output_dir)
    elif dataset_name.lower() in ["voc", "voc0712"]:
        prepare_voc0712(output_dir=output_dir)
    elif dataset_name.lower() in ["voc07", "voc2007"]:
        prepare_voc07(output_dir=output_dir)
    elif dataset_name.lower() in ["voc12", "voc2012"]:
        prepare_voc12(output_dir=output_dir)
    elif dataset_name.lower() in ["watercolor"]:
        prepare_watercolor(output_dir=output_dir)
    elif dataset_name.lower() in ["pothole"]:
        prepare_pothole(output_dir=output_dir)
    else:
        raise ValueError(f"This dataset name is not supported: {dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str)
    parser.add_argument("-o", "--output_dir", default="./", type=str)
    args = parser.parse_args()

    main(args.dataset_name, args.output_dir)
