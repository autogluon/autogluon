import autogluon as ag
from autogluon import ObjectDetection as task
import os 
import argparse
import logging

# meta info for each dataset. { name: (url, index_file_name_trainval, index_file_name_test), ...}
dataset_dict = {
    'clipart': ('http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/cross_domain_detection/datasets/clipart.zip',
                'train', 'test', None),
    'watercolor': ('http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/cross_domain_detection/datasets/watercolor.zip',
                  'instance_level_annotated', 'test', None),
    'comic': ('http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/cross_domain_detection/datasets/comic.zip',
              'instance_level_annotated', 'test', None),
    'tiny_motorbike': ('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip',
                       'trainval', 'test', ('motorbike',))
}

def get_dataset(args):
    # built-in dataset (voc)
    if 'voc' in args.dataset_name:
        logging.info('Please follow this instruction to download dataset: \
            https://gluon-cv.mxnet.io/build/examples_datasets/pascal_voc.html#sphx-glr-build-examples-datasets-pascal-voc-py ')
        train_dataset = task.Dataset(name=args.dataset_name)
        test_dataset = task.Dataset(name=args.dataset_name, Train=False)
        return (train_dataset, test_dataset)        

    # custom datset. 
    if args.dataset_name in dataset_dict: 
        url, index_file_name_trainval, index_file_name_test, classes, \
             = dataset_dict[args.dataset_name]

        data_root = os.path.join(args.dataset_root, args.dataset_name)
        if not args.no_redownload:
            root = args.dataset_root
            filename_zip = ag.download(url, path=root)
            filename = ag.unzip(filename_zip, root=root)
            data_root = os.path.join(root, filename)
    else:
        logging.info("This dataset is not in dataset_dict. It should be downloaded before running this script.")
        index_file_name_trainval = args.index_file_name_trainval
        index_file_name_test = args.index_file_name_test
        classes = args.classes
        
    train_dataset = task.Dataset(data_root, index_file_name=index_file_name_trainval, classes=classes)
    test_dataset = task.Dataset(data_root, index_file_name=index_file_name_test, classes=classes, Train=False)

    return (train_dataset, test_dataset)        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark for object detection')
    parser.add_argument('--dataset-name', type=str, default='voc', help="dataset name")
    parser.add_argument('--dataset-root', type=str, default='./', help="root path to the downloaded dataset, only for custom datastet")
    parser.add_argument('--dataset-format', type=str, default='voc', help="dataset format")
    parser.add_argument('--index-file-name-trainval', type=str, default='', help="name of txt file which contains images for training and validation ")
    parser.add_argument('--index-file-name-test', type=str, default='', help="name of txt file which contains images for testing")
    parser.add_argument('--classes', type=tuple, default=None, help="classes for custom classes")
    parser.add_argument('--no-redownload',  action='store_true', help="whether need to re-download dataset")
    args = parser.parse_args()
    logging.info('args: {}'.format(args))

    dataset_train, dataset_test = get_dataset(args) 

    time_limits = 5*60*60 # 5 days
    epochs = 1
    detector = task.fit(dataset_train,
                        num_trials=30,
                        epochs=epochs,
                        net=ag.Categorical('darknet53', 'mobilenet1.0'),
                        lr=ag.Categorical(1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5),
                        data_shape=ag.Categorical(320, 416),
                        ngpus_per_trial=1,
                        batch_size=8,
                        lr_decay_epoch=ag.Categorical('80,90','85,95'),
                        warmup_epochs=ag.Int(1, 10),
                        syncbn=ag.Bool(), 
                        label_smooth=ag.Bool(), 
                        time_limits=time_limits,
                        dist_ip_addrs = [])

    test_map = detector.evaluate(dataset_test)
    print("mAP on test dataset: {}".format(test_map[1][1]))


