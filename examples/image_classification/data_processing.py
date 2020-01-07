import csv
import os
import pandas as pd
import shutil
import string
from gluoncv.utils import makedirs
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for different kaggle competitions.')
    parser.add_argument('--data-dir', type=str, default='/home/ubuntu/workspace/autogluon_kaggle/examples/image_classification/data/',
                        help='training and validation pictures to use.')
    parser.add_argument('--dataset', type=str, default='dog',
                        help='the kaggle competition')
    opt = parser.parse_args()
    return opt
opt = parse_args()

if opt.dataset == 'dog':

    csvfile = "labels.csv"
    pic_path = "images_all/"
    train_path = "images/"
    csvfile = os.path.join(opt.data_dir,'dog-breed-identification',csvfile)
    pic_path = os.path.join(opt.data_dir,'dog-breed-identification',pic_path)
    train_path = os.path.join(opt.data_dir,'dog-breed-identification',train_path)

    csvfile = open(csvfile, 'r')
    data = []
    for line in csvfile:
        data.append(list(line.strip().split(',')))
    for i in range(len(data)):
        if i == 0:
            continue
        if i >= 1:
            cl = data[i][1]
            name = data[i][0]
            path = pic_path + str(name) + '.jpg'
            isExists = os.path.exists(path)
            if (isExists):
                if not os.path.exists(train_path + cl):
                    os.makedirs(train_path + cl)
                newpath = train_path + cl + '/' + str(name) + '.jpg'
                shutil.copyfile(path, newpath)
                print(str(name) + ',success')
            else:
                print(str(name) + ",not here")
elif opt.dataset == 'aerial':
    csvfile = "train.csv"
    pic_path = "images_all/"
    train_path = "images/"

    csvfile = os.path.join(opt.data_dir,'aerial-cactus-identification',csvfile)
    pic_path = os.path.join(opt.data_dir,'aerial-cactus-identification',pic_path)
    train_path = os.path.join(opt.data_dir,'aerial-cactus-identification',train_path)

    csvfile = open(csvfile, 'r')
    data = []
    for line in csvfile:
        data.append(list(line.strip().split(',')))
    for i in range(len(data)):
        if i == 0:
            continue
        if i >= 1:
            cl = data[i][1]
            name = data[i][0]
            path = pic_path + str(name)
            isExists = os.path.exists(path)
            if (isExists):
                if not os.path.exists(train_path + cl):
                    os.makedirs(train_path + cl)
                newpath = train_path + cl + '/' + str(name)
                shutil.copyfile(path, newpath)
                print(str(name) + ',success')
            else:
                print(str(name) + ",not here")

##
elif opt.dataset == 'fisheries_Monitoring':
    csvfile = os.path.join(opt.data_dir, opt.dataset, 'auto_5_30_fish.csv')
    df = pd.read_csv(csvfile)
    def get_name(name):
        if name.startswith('image'):
            name = 'test_stg2/' + name
        return name
    df['image'] = df['image'].apply(get_name)
    df.to_csv(csvfile.replace('auto_5_30_fish', 'auto_5_30_fish_add'), index=False)



