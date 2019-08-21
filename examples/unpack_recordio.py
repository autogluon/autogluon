import os
import numpy as np
import mxnet as mx
import autogluon as ag
from PIL import Image
from tqdm import tqdm

def get_dataset_meta(dataset, basedir='./datasets'):
    if dataset.lower() == 'apparel':
        num_classes = 18
        rec_train = os.path.join(basedir, 'Apparel_train.rec')
        rec_train_idx = os.path.join(basedir, 'Apparel_train.idx')
        rec_val = os.path.join(basedir, 'Apparel_test.rec')
        rec_val_idx = os.path.join(basedir, 'Apparel_test.idx')
    elif dataset.lower() == 'footwear':
        num_classes = 19
        rec_train = os.path.join(basedir, 'Footwear_train.rec')
        rec_train_idx = os.path.join(basedir, 'Footwear_train.idx')
        rec_val = os.path.join(basedir, 'Footwear_test.rec')
        rec_val_idx = os.path.join(basedir, 'Footwear_test.idx')
    elif dataset.lower() == 'landmarks':
        num_classes = 20
        rec_train = os.path.join(basedir, 'Landmarks_train.rec')
        rec_train_idx = os.path.join(basedir, 'Landmarks_train.idx')
        rec_val = os.path.join(basedir, 'Landmarks_test.rec')
        rec_val_idx = os.path.join(basedir, 'Landmarks_test.idx')
    elif dataset.lower() == 'weapons':
        num_classes = 11
        rec_train = os.path.join(basedir, 'Weapons_train.rec')
        rec_train_idx = os.path.join(basedir, 'Weapons_train.idx')
        rec_val = os.path.join(basedir, 'Weapons_test.rec')
        rec_val_idx = os.path.join(basedir, 'Weapons_test.idx')
    else:
        raise NotImplemented
    return num_classes, rec_train, rec_train_idx, rec_val, rec_val_idx


def unpack_recordio(dataset, rec_train, rec_train_idx, training=True):
    data_iter = mx.io.ImageRecordIter(
        path_imgrec=rec_train,
        path_imgidx=rec_train_idx,
        data_shape=(3, 256, 256),
        batch_size=1,
        resize=320)

    tbar = tqdm(enumerate(data_iter))
    for i, batch in tbar:
        data = batch.data[0].squeeze().swapaxes(0, 2).swapaxes(0, 1).astype('uint8').asnumpy()
        label = batch.label[0].asscalar()
        
        if training:
            image_path = '{dataset}/train/{label}/{i}.jpg'.format(dataset=dataset, label=label, i=i)
        else:
            image_path = '{dataset}/val/{label}/{i}.jpg'.format(dataset=dataset, label=label, i=i)
        dir_name = os.path.dirname(image_path)
        if not os.path.isdir(dir_name):
            print('making dir {}'.format(dir_name))
            ag.mkdir(dir_name)
        img = Image.fromarray(data)
        tbar.set_description(image_path)
        img.save(image_path)


if __name__ == '__main__':
    dataset = 'apparel'
    _, rec_train, rec_train_idx, rec_val, rec_val_idx = get_dataset_meta(dataset)
    unpack_recordio(dataset, rec_train, rec_train_idx, training=True)
    unpack_recordio(dataset, rec_val, rec_val_idx, training=False)
