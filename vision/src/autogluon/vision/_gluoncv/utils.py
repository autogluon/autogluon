"""Utils for auto tasks"""
import copy
import warnings
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


class ConfigDict(dict):
    """The view of a config dict where keys can be accessed like attribute, it also prevents
    naive modifications to the key-values.
    Parameters
    ----------
    config : dict
        The configuration dict.
    Attributes
    ----------
    __dict__ : type
        The internal config as a `__dict__`.
    """
    MARKER = object()
    def __init__(self, value=None):
        super(ConfigDict, self).__init__()
        self.__dict__['_freeze'] = False
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict, given {}'.format(type(value)))
        self.freeze()

    def freeze(self):
        self.__dict__['_freeze'] = True

    def is_frozen(self):
        return self.__dict__['_freeze']

    def unfreeze(self):
        self.__dict__['_freeze'] = False

    def __setitem__(self, key, value):
        if self.__dict__.get('_freeze', False):
            msg = ('You are trying to modify the config to "{}={}" after initialization, '
                   ' this may result in unpredictable behaviour'.format(key, value))
            warnings.warn(msg)
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        super(ConfigDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, ConfigDict.MARKER)
        if found is ConfigDict.MARKER:
            if self.__dict__['_freeze']:
                raise KeyError(key)
            found = ConfigDict()
            super(ConfigDict, self).__setitem__(key, found)
        if isinstance(found, ConfigDict):
            found.__dict__['_freeze'] = self.__dict__['_freeze']
        return found

    def __setstate__(self, state):
        vars(self).update(state)

    def __getstate__(self):
        return vars(self)

    __setattr__, __getattr__ = __setitem__, __getitem__

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

def get_recursively(search_dict, field):
    """
    Takes a dict with nested dicts,
    and searches all dicts for a key of the field
    provided.
    """
    fields_found = []

    for key, value in search_dict.items():

        if key == field:
            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_recursively(value, field)
            for result in results:
                fields_found.append(result)

    return fields_found

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

def config_to_nested_v0(config):
    """Convert config to nested version"""
    if 'meta_arch' not in config:
        if config['estimator'] == SSDEstimator:
            config['meta_arch'] = 'ssd'
        elif config['estimator'] == FasterRCNNEstimator:
            config['meta_arch'] = 'faster_rcnn'
        elif config['estimator'] == YOLOv3Estimator:
            config['meta_arch'] = 'yolo3'
        elif config['estimator'] == CenterNetEstimator:
            config['meta_arch'] = 'center_net'
        elif config['estimator'] == ImageClassificationEstimator:
            config['meta_arch'] = 'img_cls'
        else:
            config['meta_arch'] = None
    else:
        pass

    if config['meta_arch'] == 'ssd':
        config_mapping = {
            'ssd': ['backbone', 'data_shape', 'filters', 'sizes', 'ratios', 'steps', 'syncbn',
                    'amp', 'custom_model'],
            'train': ['batch_size', 'start_epoch', 'epochs', 'lr', 'lr_decay', 'lr_decay_epoch',
                      'momentum', 'wd', 'log_interval', 'seed', 'dali'],
            'validation': ['val_interval']
        }
    elif config['meta_arch'] == 'faster_rcnn':
        config_mapping = {
            'faster_rcnn': ['backbone', 'nms_thresh', 'nms_topk', 'roi_mode', 'roi_size', 'strides', 'clip',
                            'anchor_base_size', 'anchor_aspect_ratio', 'anchor_scales', 'anchor_alloc_size',
                            'rpn_channel', 'rpn_nms_thresh', 'max_num_gt', 'norm_layer', 'use_fpn', 'num_fpn_filters',
                            'num_box_head_conv', 'num_box_head_conv_filters', 'num_box_head_dense_filters',
                            'image_short', 'image_max_size', 'custom_model', 'amp', 'static_alloc',
                            'disable_hybridization'],
            'train': ['pretrained_base', 'batch_size', 'start_epoch', 'epochs', 'lr', 'lr_decay',
                      'lr_decay_epoch', 'lr_mode', 'lr_warmup', 'lr_warmup_factor', 'momentum', 'wd',
                      'rpn_train_pre_nms', 'rpn_train_post_nms', 'rpn_smoothl1_rho', 'rpn_min_size',
                      'rcnn_num_samples', 'rcnn_pos_iou_thresh', 'rcnn_pos_ratio', 'rcnn_smoothl1_rho',
                      'log_interval', 'seed', 'verbose', 'mixup', 'no_mixup_epochs', 'executor_threads'],
            'validation': ['rpn_test_pre_nms', 'rpn_test_post_nms', 'val_interval']
        }
    elif config['meta_arch'] == 'yolo3':
        config_mapping = {
            'yolo3': ['backbone', 'filters', 'anchors', 'strides', 'data_shape', 'syncbn', 'no_random_shape',
                      'amp', 'custom_model'],
            'train': ['batch_size', 'epochs', 'start_epoch', 'lr', 'lr_mode', 'lr_decay', 'lr_decay_period',
                      'lr_decay_epoch', 'warmup_lr', 'warmup_epochs', 'momentum', 'wd', 'log_interval',
                      'seed', 'num_samples', 'no_wd', 'mixup', 'no_mixup_epochs', 'label_smooth'],
            'validation': ['val_interval']
        }
    elif config['meta_arch'] == 'center_net':
        config_mapping = {
            'center_net': ['base_network', 'heads', 'scale', 'topk', 'root', 'wh_weight', 'center_reg_weight',
                           'data_shape'],
            'train': ['gpus', 'pretrained_base', 'batch_size', 'epochs', 'lr', 'lr_decay', 'lr_decay_epoch',
                      'lr_mode', 'warmup_lr', 'warmup_epochs', 'num_workers', 'resume',
                      'start_epoch', 'momentum', 'wd', 'save_interval', 'log_interval'],
            'validation': ['flip_test', 'nms_thresh', 'nms_topk', 'post_nms', 'num_workers',
                           'batch_size', 'interval']
        }
    elif config['meta_arch'] == 'img_cls':
        config_mapping = {
            'img_cls': ['model', 'use_pretrained', 'use_gn', 'batch_norm', 'use_se', 'last_gamma'],
            'train': ['gpus', 'num_workers', 'batch_size', 'epochs', 'start_epoch', 'lr', 'lr_mode',
                      'lr_decay', 'lr_decay_period',
                      'lr_decay_epoch', 'warmup_lr', 'warmup_epochs', 'momentum', 'wd', 'log_interval',
                      'seed', 'num_samples', 'no_wd', 'mixup', 'no_mixup_epochs', 'label_smooth'],
            'validation': []
        }
    else:
        raise NotImplementedError('%s is not implemented.' % config['meta_arch'])

    nested_config = {}

    for k, v in config.items():
        if k in config_mapping[config['meta_arch']]:
            if config['meta_arch'] not in nested_config:
                nested_config[config['meta_arch']] = {}
            nested_config[config['meta_arch']].update({k: v})
        elif k in config_mapping['train']:
            if 'train' not in nested_config:
                nested_config['train'] = {}
            nested_config['train'].update({k: v})
        elif k in config_mapping['validation']:
            if 'validation' not in nested_config:
                nested_config['validation'] = {}
            nested_config['validation'].update({k: v})
        else:
            nested_config.update({k: v})

    return nested_config

def recursive_update(total_config, config):
    """update config recursively"""
    for k, v in config.items():
        if isinstance(v, dict):
            if k not in total_config:
                total_config[k] = {}
            recursive_update(total_config[k], v)
        else:
            total_config[k] = v

def config_to_space(config):
    """Convert config to ag.space"""
    space = ag.Dict()
    for k, v in config.items():
        if isinstance(v, dict):
            if k not in space:
                space[k] = ag.Dict()
            space[k] = config_to_space(v)
        else:
            space[k] = v
    return space

def auto_args(config, estimators):
    """
    Merge user defined config to estimator default config, and convert to search space
    Parameters
    ----------
    config: <class 'dict'>
    Returns
    -------
    ag_space: <class 'autogluon.core.space.Dict'>
    """
    total_config = {}

    # estimator default config
    if not isinstance(estimators, (tuple, list)):
        estimators = [estimators]
    for estimator in estimators:
        assert issubclass(estimator, BaseEstimator), estimator
        default_config = copy.deepcopy(estimator._default_config)  # <class 'dict'>
        recursive_update(total_config, default_config)

    # user defined config
    nested_config = config_to_nested(config)
    recursive_update(total_config, nested_config)

    # convert to search space
    ag_space = config_to_space(total_config)

    return ag_space
