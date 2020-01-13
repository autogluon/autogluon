import os
import math
import pickle
import mxnet as mx
import matplotlib.pyplot as plt
from mxnet.gluon.data.vision import transforms

from .utils import *
from ..base.base_predictor import BasePredictor
from ...utils import save, load, tqdm

import warnings
import logging

import numpy as np
from mxnet.gluon import nn
from mxnet import gluon, init, autograd, nd
from mxnet.gluon.data.vision import transforms
import gluoncv as gcv
from gluoncv.model_zoo import get_model
from gluoncv import utils as gutils

from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils import LRScheduler, LRSequential

from gluoncv.data.transforms import presets
import gluoncv as gcv

__all__ = ['Detector']

class Detector(BasePredictor):
    """
    Trained Object Detector returned by `task.fit()`
    """
    def __init__(self, model, results, scheduler_checkpoint,
                 args, **kwargs):
        self.model = model
        self.results = self._format_results(results)
        self.scheduler_checkpoint = scheduler_checkpoint
        self.args = args


    def evaluate(self, dataset, ctx=[mx.cpu()]):
        """Evaluate performance of this object detector's predictions on test data.
         
         Parameters
         ----------
        dataset: `Dataset`
            Test dataset (must be in the same format as training data previously provided to fit).
        ctx : List of `mxnet.context` elements.
            Determines whether to use CPU or GPU(s), options include: `[mx.cpu()]` or `[mx.gpu()]`.
        """
        args = self.args
        net = self.model
        batch_size = args.batch_size * max(len(ctx), 1)

        def _get_dataloader(net, test_dataset, data_shape, batch_size, num_workers, args):
            """Get dataloader."""
            width, height = data_shape, data_shape
            val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
            test_loader = gluon.data.DataLoader(
                test_dataset.transform(YOLO3DefaultValTransform(width, height)),
                batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
            return test_loader
        
        def _validate(net, val_data, ctx, eval_metric):
            """Test on validation dataset."""
            eval_metric.reset()
            # set nms threshold and topk constraint
            net.set_nms(nms_thresh=0.45, nms_topk=400)
            mx.nd.waitall()
            net.hybridize()
            for batch in val_data:
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
                det_bboxes = []
                det_ids = []
                det_scores = []
                gt_bboxes = []
                gt_ids = []
                gt_difficults = []
                for x, y in zip(data, label):
                    # get prediction results
                    ids, scores, bboxes = net(x)
                    det_ids.append(ids)
                    det_scores.append(scores)
                    # clip to image size
                    det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                    # split ground truths
                    gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                    gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                    gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

                # update metric
                eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
            return eval_metric.get()

        if isinstance(dataset, AutoGluonObject):
            dataset = dataset.init()
        test_dataset, eval_metric = dataset.get_dataset_and_metric()
        test_data = _get_dataloader(net, test_dataset, args.data_shape, args.batch_size, args.num_workers, args)
        return _validate(net, test_data, ctx, eval_metric)
    
    @staticmethod
    def _format_results(results):
        def _merge_scheduler_history(training_history, config_history, reward_attr):
            trial_info = {}
            for tid, config in config_history.items():
                trial_info[tid] = {}
                trial_info[tid]['config'] = config
                if tid in training_history:
                    trial_info[tid]['history'] = training_history[tid]
                    trial_info[tid]['metadata'] = {}

                    if len(training_history[tid]) > 0 and reward_attr in training_history[tid][-1]:
                        last_history = training_history[tid][-1]
                        trial_info[tid][reward_attr] = last_history.pop(reward_attr)
                        trial_info[tid]['metadata'].update(last_history)
            return trial_info

        training_history = results.pop('training_history')
        config_history = results.pop('config_history')
        results['trial_info'] = _merge_scheduler_history(training_history, config_history,
                                                              results['reward_attr'])
        results[results['reward_attr']] = results.pop('best_reward')
        results['search_space'] = results['metadata'].pop('search_space')
        results['search_strategy'] = results['metadata'].pop('search_strategy')
        return results
    
    def predict(self, X, input_size=224, thresh=0.15, plot=True):
        """ Use this object detector to make predictions on test data.
        
        Parameters
        ----------
        X : Test data with image(s) to make predictions for.
        input_size : int
            Size of images in test data (pixels).
        thresh : float
            Confidence Threshold above which detector outputs bounding box for object.
        plot : bool
            Whether or not to plot the bounding box of detected objects on top of the original images.
        
        Returns
        -------
        Tuple containing the class-IDs of detected objects, the confidence-scores associated with 
        these detectiions, and the corresponding predicted bounding box locations.
        """
        net = self.model
        net.set_nms(0.45, 200)
        net.collect_params().reset_ctx(ctx = mx.cpu())

        x, img = presets.yolo.load_test(X, short=512)
        ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]

        if plot:
            ax = gcv.utils.viz.plot_bbox(img, bboxes, scores, ids, thresh=thresh,
                                        class_names=net.classes, ax=None)
            plt.show()
        return ids, scores, bboxes
        

    def load(cls, checkpoint):
        raise NotImplemented
    
    def save(self, checkpoint):
        raise NotImplemented

    
    def predict_proba(self, X):
        raise NotImplemented
    
    def _save_model(self, *args, **kwargs):
        raise NotImplemented

    def evaluate_predictions(self, y_true, y_pred):
        raise NotImplemented
        

