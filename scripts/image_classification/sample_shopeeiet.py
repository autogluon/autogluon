from autogluon.utils.try_import import *
try_import_cv2()

import os, argparse, shutil, cv2

# run `sh download_shopeeiet.sh` first

def parse_opts():
    parser = argparse.ArgumentParser(description='Preparing Shopee-IET Dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, required=True,
                        help='directory for the original data folder')
    parser.add_argument('--split', type=int, default=9,
                        help='# samples of train/# samples of val')
    opts = parser.parse_args()
    return opts

# Preparation
opts = parse_opts()

path = opts.data
split = opts.split

def _generate_image_label(split):
    train_image_list = []
    val_image_list = []
    labels = []
    for label in os.listdir(os.path.join(path, 'images')):
        labels.append(label)
        i = 0
        for img in os.listdir(os.path.join(path, 'images', label)):
            if i % split == 0 and i != 0:
                val_image_list.append('%s/%s' % (label,img))
            else:
                train_image_list.append('%s/%s' % (label,img))
            i += 1
    labels = sorted(labels)
    return train_image_list, val_image_list, labels

_, _, labels = _generate_image_label(split)

# Create directories
src_train_path = os.path.join(path, 'train')
src_val_path = os.path.join(path, 'val')
sample_train_path = os.path.join(path, 'train_sample')
sample_val_path = os.path.join(path, 'val_sample')
os.makedirs(sample_train_path)
os.makedirs(sample_val_path)

selected_classes = ['BabyPants', 'BabyShirt', 'womencasualshoes', 'womenchiffontop']
for l in labels:
    if l not in selected_classes:
        continue
    os.makedirs(os.path.join(sample_train_path, l))
    os.makedirs(os.path.join(sample_val_path, l))

print(labels)

# Copy files to corresponding directory
for label in labels:
    if label not in selected_classes:
        continue
    count = 0
    for img in os.listdir(os.path.join(src_train_path, label)):
        if count == 200:
            break
        shutil.copy(os.path.join(src_train_path, label, img),
                    os.path.join(sample_train_path, label, img))
        count += 1

# Copy files to corresponding directory
for label in labels:
    if label not in selected_classes:
        continue
    count = 0
    for img in os.listdir(os.path.join(src_val_path, label)):
        if count == 20:
            break
        shutil.copy(os.path.join(src_val_path, label, img),
                    os.path.join(sample_val_path, label, img))
        count += 1
