import argparse
import logging
import os

import mxnet as mx

import autogluon.core as ag
from autogluon.vision.object_detection import ObjectDetector
from autogluon.core.scheduler import get_gpu_count

# meta info for each dataset. { name: (url, index_file_name_trainval, index_file_name_test), ...}
dataset_dict = {
    'clipart': (
        'http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/cross_domain_detection/datasets/clipart.zip',
        'train', 'test', None),
    'watercolor': (
        'http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/cross_domain_detection/datasets/watercolor.zip',
        'instance_level_annotated', 'test', None),
    'comic': (
        'http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/cross_domain_detection/datasets/comic.zip',
        'instance_level_annotated', 'test', None),
    'tiny_motorbike': ('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip',
                       'trainval', 'test', ('motorbike',))
}


def get_dataset(args):
    # built-in dataset (voc)
    if 'voc' in args.dataset_name:
        logging.info('Please follow this instruction to download dataset: \
            https://gluon-cv.mxnet.io/build/examples_datasets/pascal_voc.html#sphx-glr-build-examples-datasets-pascal-voc-py ')
        root = os.path.expanduser('~/.mxnet/datasets/voc/VOC2007')
        train_dataset = ObjectDetector.Dataset.from_voc(root, splits='trainval')
        test_dataset = ObjectDetector.Dataset.from_voc(root, splits='test')
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
        logging.info(
            "This dataset is not in dataset_dict. It should be downloaded before running this script.")
        index_file_name_trainval = args.index_file_name_trainval
        index_file_name_test = args.index_file_name_test
        data_root = args.data_root

    train_dataset = ObjectDetector.Dataset.from_voc(data_root, splits=index_file_name_trainval)
    test_dataset = ObjectDetector.Dataset.from_voc(data_root, splits=index_file_name_test)

    return (train_dataset, test_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark for object detection')
    parser.add_argument('--dataset-name', type=str, default='voc', help="dataset name")
    parser.add_argument('--dataset-root', type=str, default='./',
                        help="root path to the downloaded dataset, only for custom datastet")
    parser.add_argument('--dataset-format', type=str, default='voc', help="dataset format")
    parser.add_argument('--index-file-name-trainval', type=str, default='',
                        help="name of txt file which contains images for training and validation ")
    parser.add_argument('--index-file-name-test', type=str, default='',
                        help="name of txt file which contains images for testing")
    parser.add_argument('--no-redownload', action='store_true',
                        help="whether need to re-download dataset")
    parser.add_argument('--num-trials', type=int, default=3,
                        help="the HPO trials to perform")
    parser.add_argument('--meta-arch', type=str, default='yolo3', choices=['yolo3', 'faster_rcnn'],
                        help="Meta architecture of the model")

    args = parser.parse_args()
    logging.info('args: {}'.format(args))

    dataset_train, dataset_test = get_dataset(args)

    time_limit = 5 * 24 * 60 * 60  # 5 days
    epochs = 20
    if args.meta_arch == 'yolo3':
        transfer = None if ('voc' in args.dataset_name) or ('coco' in args.dataset_name) else \
            ag.Categorical('yolo3_darknet53_coco', 'yolo3_mobilenet1.0_coco')
        hyperparameters = {
            'estimator': args.meta_arch,
            'lr': ag.Categorical(1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5),
            'data_shape': ag.Categorical(320, 416),
            'batch_size': 16,
            'lr_decay_epoch': ag.Categorical([80,90], [85,95]),
            'warmup_epochs': ag.Int(1, 10), 'warmup_iters': ag.Int(250, 1000),
            'wd': ag.Categorical(1e-4, 5e-4, 2.5e-4), 'syncbn': ag.Categorical(True, False),
            'epochs': epochs,
            'transfer': transfer
        }
        kwargs = {'num_trials': args.num_trials,
                  'time_limit': time_limit,
                  'dist_ip_addrs': [],
                  'nthreads_per_trial': 16,
                  'ngpus_per_trial': 8,
                  'hyperparameters': hyperparameters}
    elif args.meta_arch == 'faster_rcnn':
        transfer = None if ('voc' in args.dataset_name) or ('coco' in args.dataset_name) else \
            ag.Categorical('faster_rcnn_fpn_resnet101_v1d_coco', 'faster_rcnn_resnet50_v1b_coco')
        hyperparameters = {
            'estimator': args.meta_arch,
            'lr': ag.Categorical(0.02, 0.01, 0.005, 0.002, 2e-4, 5e-4),
            'data_shape': (640, 800),
            'lr_decay_epoch': ag.Categorical([24,28], [35], [50,55], [40], [45], [55],
                                             [30, 35], [20]),
            'warmup_iters': ag.Int(5, 500),
            'wd': ag.Categorical(1e-4, 5e-4, 2.5e-4), 'syncbn': True,
            'label_smooth': False,
            'epochs': ag.Categorical(30, 40, 50, 60),
            'transfer': transfer
        }
        kwargs = {'num_trials': args.num_trials,
                  'nthreads_per_trial': 16,
                  'ngpus_per_trial': 8,
                  'time_limit': time_limit,
                  'dist_ip_addrs': [],
                  'hyperparameters': hyperparameters}
    else:
        raise NotImplementedError('%s is not implemented.', args.meta_arch)
    detector = ObjectDetector()
    detector.fit(dataset_train, **kwargs)
    test_map = detector.evaluate(dataset_test)
    print("mAP on test dataset: {}".format(test_map[-1][-1]))
    print(test_map)
    detector.save('final_model.model')
