#  from Gluon CV script:
#  https://github.com/dmlc/gluon-cv/blob/master/scripts/classification/finetune/prepare_minc.py

import os, argparse, shutil

def parse_opts():
    parser = argparse.ArgumentParser(description='Preparing MINC 2500 Dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, required=True,
                        help='directory for the original data folder')
    parser.add_argument('--split', type=int, default=1,
                        help='an integer in [1, 5] to specify the train/val/test split file')
    opts = parser.parse_args()
    return opts

# Preparation
opts = parse_opts()

path = opts.data
split = opts.split

# Read files
train_images_file = os.path.join(path, 'labels/train' + str(split) + '.txt')
with open(train_images_file, 'r') as f:
    train_images = f.readlines()

val_images_file = os.path.join(path, 'labels/validate' + str(split) + '.txt')
with open(val_images_file, 'r') as f:
    val_images = f.readlines()

test_images_file = os.path.join(path, 'labels/test' + str(split) + '.txt')
with open(test_images_file, 'r') as f:
    test_images = f.readlines()

# Create directories
src_path = os.path.join(path, 'images')
train_path = os.path.join(path, 'train')
val_path = os.path.join(path, 'val')
test_path = os.path.join(path, 'test')
os.makedirs(train_path)
os.makedirs(val_path)
os.makedirs(test_path)

labels = sorted(os.listdir(src_path))

for l in labels:
    os.makedirs(os.path.join(train_path, l))
    os.makedirs(os.path.join(val_path, l))
    os.makedirs(os.path.join(test_path, l))

# Copy files to corresponding directory
for im in train_images:
    im_path = im.replace('images/', '').strip('\n')
    shutil.copy(os.path.join(src_path, im_path),
                os.path.join(train_path, im_path))

for im in val_images:
    im_path = im.replace('images/', '').strip('\n')
    shutil.copy(os.path.join(src_path, im_path),
                os.path.join(val_path, im_path))

for im in test_images:
    im_path = im.replace('images/', '').strip('\n')
    shutil.copy(os.path.join(src_path, im_path),
                os.path.join(test_path, im_path))
