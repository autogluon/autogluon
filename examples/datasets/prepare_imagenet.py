"""Prepare the ImageNet dataset"""
import os
import argparse
import tarfile
import pickle
import gzip
import subprocess
from tqdm import tqdm
from autogluon.core.utils import check_sha1, download, mkdir

_TARGET_DIR = os.path.expanduser('~/.autogluon/datasets/imagenet')
_TRAIN_TAR = 'ILSVRC2012_img_train.tar'
_TRAIN_TAR_SHA1 = '43eda4fe35c1705d6606a6a7a633bc965d194284'
_VAL_TAR = 'ILSVRC2012_img_val.tar'
_VAL_TAR_SHA1 = '5f3f73da3395154b60528b2b2a2caf2374f5f178'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Setup the ImageNet dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', required=True,
                        help="The directory that contains downloaded tar files")
    parser.add_argument('--target-dir', default=_TARGET_DIR,
                        help="The directory to store extracted images")
    parser.add_argument('--checksum', action='store_true',
                        help="If check integrity before extracting.")
    parser.add_argument('--with-rec', action='store_true',
                        help="If build image record files.")
    parser.add_argument('--num-thread', type=int, default=1,
                        help="Number of threads to use when building image record file.")
    args = parser.parse_args()
    return args

def check_file(filename, checksum, sha1):
    if not os.path.exists(filename):
        raise ValueError('File not found: '+filename)
    if checksum and not check_sha1(filename, sha1):
        raise ValueError('Corrupted file: '+filename)

def build_rec_process(img_dir, train=False, num_thread=1):
    rec_dir = os.path.abspath(os.path.join(img_dir, '../rec'))
    mkdir(rec_dir)
    prefix = 'train' if train else 'val'
    print('Building ImageRecord file for ' + prefix + ' ...')
    to_path = rec_dir

    # download lst file and im2rec script
    script_path = os.path.join(rec_dir, 'im2rec.py')
    script_url = 'https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py'
    download(script_url, script_path)

    # execution
    import sys
    cmd = [
        sys.executable,
        script_path,
        rec_dir,
        img_dir,
        '--recursive',
        '--pass-through',
        '--pack-label',
        '--num-thread',
        str(num_thread)
    ]
    subprocess.call(cmd)
    os.remove(script_path)
    print('ImageRecord file for ' + prefix + ' has been built!')

def extract_train(tar_fname, target_dir, with_rec=False, num_thread=1):
    mkdir(target_dir)
    with tarfile.open(tar_fname) as tar:
        print("Extracting "+tar_fname+"...")
        # extract each class one-by-one
        pbar = tqdm(total=len(tar.getnames()))
        for class_tar in tar:
            pbar.set_description('Extract '+class_tar.name)
            tar.extract(class_tar, target_dir)
            class_fname = os.path.join(target_dir, class_tar.name)
            class_dir = os.path.splitext(class_fname)[0]
            os.mkdir(class_dir)
            with tarfile.open(class_fname) as f:
                f.extractall(class_dir)
            os.remove(class_fname)
            pbar.update(1)
        pbar.close()
    if with_rec:
        build_rec_process(target_dir, True, num_thread)

def extract_val(tar_fname, target_dir, with_rec=False, num_thread=1):
    mkdir(target_dir)
    print('Extracting ' + tar_fname)
    with tarfile.open(tar_fname) as tar:
        tar.extractall(target_dir)
    # build rec file before images are moved into subfolders
    if with_rec:
        build_rec_process(target_dir, False, num_thread)
    # move images to proper subfolders
    val_maps_file = os.path.join(os.path.dirname(__file__), 'imagenet_val_maps.pklz')
    download('https://gluon-cv.mxnet.io/_downloads/f5c3f5262b5968d15a687bf7bd73db68/imagenet_val_maps.pklz', val_maps_file)
    with gzip.open(val_maps_file, 'rb') as f:
        dirs, mappings = pickle.load(f)
    for d in dirs:
        os.makedirs(os.path.join(target_dir, d))
    for m in mappings:
        os.rename(os.path.join(target_dir, m[0]), os.path.join(target_dir, m[1], m[0]))

def main():
    args = parse_args()

    target_dir = os.path.expanduser(args.target_dir)
    if os.path.exists(target_dir):
        raise ValueError('Target dir ['+target_dir+'] exists. Remove it first')

    download_dir = os.path.expanduser(args.download_dir)
    train_tar_fname = os.path.join(download_dir, _TRAIN_TAR)
    check_file(train_tar_fname, args.checksum, _TRAIN_TAR_SHA1)
    val_tar_fname = os.path.join(download_dir, _VAL_TAR)
    check_file(val_tar_fname, args.checksum, _VAL_TAR_SHA1)

    build_rec = args.with_rec
    if build_rec:
        os.makedirs(os.path.join(target_dir, 'rec'))
    extract_train(train_tar_fname, os.path.join(target_dir, 'train'), build_rec, args.num_thread)
    extract_val(val_tar_fname, os.path.join(target_dir, 'val'), build_rec, args.num_thread)

if __name__ == '__main__':
    main()
