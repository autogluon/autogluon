from autogluon.utils.try_import import try_import_cv2
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
    for label in os.listdir(path):
        if label == 'test':
            continue
        labels.append(label)
        i = 0
        for img in os.listdir(os.path.join(path, label)):
            if split == 0:
                train_image_list.append('%s/%s' % (label, img))
            else:
                if i % split == 0 and i != 0:
                    val_image_list.append('%s/%s' % (label,img))
                else:
                    train_image_list.append('%s/%s' % (label,img))
            i += 1
    labels = sorted(labels)
    return train_image_list, val_image_list, labels
train_image_list, val_image_list, labels = _generate_image_label(split)


# Create directories
src_path = path
train_path = os.path.join(path, 'train')
os.makedirs(train_path)
if len(val_image_list) != 0:
    val_path = os.path.join(path, 'val')
    os.makedirs(val_path)

for l in labels:
    os.makedirs(os.path.join(train_path, l))
    if len(val_image_list) != 0:
        os.makedirs(os.path.join(val_path, l))

# Copy files to corresponding directory
for img_path in train_image_list:
    img = cv2.imread(os.path.join(src_path, img_path))
    try:
        tmp = img.shape
    except AttributeError as e:
        print('train: ' + img_path)
        continue
    cv2.imwrite(os.path.join(train_path, img_path), img)

for img_path in val_image_list:
    img = cv2.imread(os.path.join(src_path, img_path))
    try:
        tmp = img.shape
    except AttributeError as e:
        print('val: ' + img_path)
        continue
    cv2.imwrite(os.path.join(val_path, img_path), img)