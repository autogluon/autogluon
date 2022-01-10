"""Utils for auto tasks"""
import numpy as np

import autogluon.core as ag

from gluoncv.auto.estimators.base_estimator import BaseEstimator
from gluoncv.auto.estimators import SSDEstimator, FasterRCNNEstimator, YOLOv3Estimator, CenterNetEstimator
from gluoncv.auto.estimators import ImageClassificationEstimator
from gluoncv.auto.data.dataset import ObjectDetectionDataset

try:
    from gluoncv import data as gdata
except ImportError:
    gdata = None


def auto_suggest(config, estimator, logger):
    """
    Automatically suggest some hyperparameters based on the dataset statistics.
    """
    # specify estimator search space
    if estimator is None:
        estimator_init = [SSDEstimator, YOLOv3Estimator, FasterRCNNEstimator, CenterNetEstimator]
        config['estimator'] = ag.Categorical(*estimator_init)
    elif isinstance(estimator, str):
        named_estimators = {
            'ssd': SSDEstimator,
            'faster_rcnn': FasterRCNNEstimator,
            'yolo3': YOLOv3Estimator,
            'center_net': CenterNetEstimator,
            'img_cls': ImageClassificationEstimator
        }
        if estimator.lower() in named_estimators:
            estimator = [named_estimators[estimator.lower()]]
        else:
            available_ests = named_estimators.keys()
            raise ValueError(f'Unknown estimator name: {estimator}, options: {available_ests}')
    elif isinstance(estimator, (tuple, list)):
        pass
    else:
        if isinstance(estimator, ag.Space):
            estimator = estimator.data
        elif isinstance(estimator, str):
            estimator = [estimator]
        for i, e in enumerate(estimator):
            if e == 'ssd':
                estimator[i] = SSDEstimator
            elif e == 'yolo3':
                estimator[i] = YOLOv3Estimator
            elif e == 'faster_rcnn':
                estimator[i] = FasterRCNNEstimator
            elif e == 'center_net':
                estimator[i] = CenterNetEstimator
        if not estimator:
            raise ValueError('Unable to determine the estimator for fit function.')
        if len(estimator) == 1:
            config['estimator'] = estimator[0]
        else:
            config['estimator'] = ag.Categorical(*estimator)

    # get dataset statistics
    # user needs to define a Dataset object "train_dataset" when using custom dataset
    train_dataset = config.get('train_dataset', None)
    try:
        if train_dataset is None:
            dataset_name = config.get('dataset', 'voc')
            dataset_root = config.get('dataset_root', '~/.mxnet/datasets/')
            if gdata is not None and dataset_name == 'voc':
                train_dataset = gdata.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
            elif gdata is not None and dataset_name == 'voc_tiny':
                train_dataset = gdata.CustomVOCDetectionBase(classes=('motorbike',),
                                                             root=dataset_root + 'tiny_motorbike',
                                                             splits=[('', 'trainval')])
            elif gdata is not None and dataset_name == 'coco':
                train_dataset = gdata.COCODetection(splits=['instances_train2017'])
            else:
                if gdata is None:
                    logger.warning('Unable to import mxnet so voc and coco formats are currently not loaded.')
        elif isinstance(train_dataset, ObjectDetectionDataset):
            train_dataset = train_dataset.to_mxnet()
        else:
            logger.info('Unknown dataset, quit auto suggestion...')
            return
    # pylint: disable=broad-except
    except Exception as e:
        logger.info(f'Unexpected error: {e}, quit auto suggestion...')
        return

    # choose 100 examples to calculate average statistics
    num_examples = 100
    image_size_list = []
    num_objects_list = []
    bbox_size_list = []
    bbox_rel_size_list = []

    for i in range(num_examples):
        train_image, train_label = train_dataset[i]

        image_height = train_image.shape[0]
        image_width = train_image.shape[1]
        image_size = image_height * image_width
        image_size_list.append(image_size)

        bounding_boxes = train_label[:, :4]
        num_objects = bounding_boxes.shape[0]
        bbox_height = bounding_boxes[:, 3] - bounding_boxes[:, 1]
        bbox_width = bounding_boxes[:, 2] - bounding_boxes[:, 0]
        bbox_size = bbox_height * bbox_width
        bbox_rel_size = bbox_size / image_size
        num_objects_list.append(num_objects)
        bbox_size_list.append(np.mean(bbox_size))
        bbox_rel_size_list.append(np.mean(bbox_rel_size))

    num_images = len(train_dataset)
    image_size = np.mean(image_size_list)
    try:
        num_classes = len(train_dataset.CLASSES)
    except AttributeError:
        num_classes = len(train_dataset.classes)
    num_objects = np.mean(num_objects_list)
    bbox_size = np.mean(bbox_size_list)
    bbox_rel_size = np.mean(bbox_rel_size_list)

    logger.info("[Printing dataset statistics]...")
    logger.info("number of training images: %d", num_images)
    logger.info("average image size: %.2f", image_size)
    logger.info("number of total object classes: %d", num_classes)
    logger.info("average number of objects in an image: %.2f", num_objects)
    logger.info("average bounding box size: %.2f", bbox_size)
    logger.info("average bounding box relative size: %.2f", bbox_rel_size)

    # specify 3 parts of config: data preprocessing, model selection, training settings
    if bbox_rel_size < 0.2 or num_objects > 5:
        suggested_estimator = [FasterRCNNEstimator]
    else:
        suggested_estimator = [SSDEstimator, YOLOv3Estimator, CenterNetEstimator]

    # specify estimator search space based on suggestion
    if estimator is None:
        estimator = suggested_estimator
        config['estimator'] = ag.Categorical(*estimator)


def config_to_nested(config):
    """Convert config to nested version"""
    estimator = config.get('estimator', None)
    transfer = config.get('transfer', None)
    # choose hyperparameters based on pretrained model in transfer learning
    if transfer:
        # choose estimator
        if transfer.startswith('ssd'):
            estimator = SSDEstimator
        elif transfer.startswith('yolo3'):
            estimator = YOLOv3Estimator
        elif transfer.startswith('faster_rcnn'):
            estimator = FasterRCNNEstimator
        elif transfer.startswith('center_net'):
            estimator = CenterNetEstimator
        else:
            estimator = ImageClassificationEstimator
        # choose base network
        transfer_list = transfer.split('_')
        if transfer_list[0] == 'ssd':
            transfer_list.pop(0)
            config['data_shape'] = int(transfer_list.pop(0))
            transfer_list.pop(-1)
        elif transfer_list[0] == 'yolo3':
            transfer_list.pop(0)
            transfer_list.pop(-1)
        else:
            transfer_list.pop(0)
            transfer_list.pop(0)
            transfer_list.pop(-1)
        config['base_network'] = '_'.join(transfer_list)
        if config['base_network'].startswith('mobilenet'):
            config['base_network'].replace('_', '.')
    elif isinstance(estimator, str):
        if estimator == 'ssd':
            estimator = SSDEstimator
        elif estimator == 'yolo3':
            estimator = YOLOv3Estimator
        elif estimator == 'faster_rcnn':
            estimator = FasterRCNNEstimator
        elif estimator == 'center_net':
            estimator = CenterNetEstimator
        elif estimator == 'img_cls':
            estimator = ImageClassificationEstimator
        else:
            raise ValueError(f'Unknown estimator: {estimator}')
    else:
        assert issubclass(estimator, BaseEstimator)

    cfg_map = estimator._default_cfg.asdict()

    def _recursive_update(config, key, value, auto_strs, auto_ints):
        for k, v in config.items():
            if k in auto_strs:
                config[k] = 'auto'
            if k in auto_ints:
                config[k] = -1
            if key == k:
                config[key] = value
            elif isinstance(v, dict):
                _recursive_update(v, key, value, auto_strs, auto_ints)

    if 'use_rec' in config:
        auto_strs = ['data_dir']
        auto_ints = []
    else:
        auto_strs = ['data_dir', 'rec_train', 'rec_train_idx', 'rec_val', 'rec_val_idx',
                     'dataset', 'dataset_root']
        auto_ints = ['num_training_samples']
    for k, v in config.items():
        _recursive_update(cfg_map, k, v, auto_strs, auto_ints)
    cfg_map['estimator'] = estimator
    return cfg_map
