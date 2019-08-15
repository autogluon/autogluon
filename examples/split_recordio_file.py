import numpy as np
import mxnet as mx
from mxnet.recordio import MXIndexedRecordIO

from transfer_learning_hp import get_dataset_meta

def train_val_split(rec_train, rec_train_idx, rec_train_split, rec_train_split_idx,
                    rec_val_split, rec_val_split_idx, ratio=0.8):
    train_record = MXIndexedRecordIO(rec_train_idx, rec_train, 'r' )
    train_split = MXIndexedRecordIO(rec_train_split_idx, rec_train_split, 'w')
    val_split = MXIndexedRecordIO(rec_val_split_idx, rec_val_split, 'w')

    idx = 0
    train_idx = 0
    val_idx = 0
    while True:
        try:
            ret = train_record.read_idx(idx)
            idx += 1
            print(idx)
            if np.random.random_sample() < ratio:
                train_split.write_idx(train_idx, ret)
                train_idx += 1
            else:
                val_split.write_idx(val_idx, ret)
                val_idx += 1
        except Exception as e:
            train_record.close()
            train_split.close()
            val_split.close()
            print('Exception', e)
            break

if __name__ == '__main__':
    datasets = ['apparel', 'footwear', 'landmarks', 'weapons']

    for dataset_name in datasets:
        _, rec_train, rec_train_idx, _, _ = get_dataset_meta(dataset_name, './', final_fit=True)
        _, rec_train_split, rec_train_split_idx, rec_val_split, rec_val_split_idx = \
                get_dataset_meta(dataset_name, './', final_fit=False)
        train_val_split(rec_train, rec_train_idx, rec_train_split, rec_train_split_idx,
                        rec_val_split, rec_val_split_idx)
