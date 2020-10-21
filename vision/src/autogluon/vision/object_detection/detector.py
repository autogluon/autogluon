from collections import OrderedDict

import cloudpickle as pkl
import gluoncv as gcv
import matplotlib.pyplot as plt
import mxnet as mx
from gluoncv.data.batchify import Tuple, Stack, Pad, Append
from gluoncv.data.transforms import presets
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultValTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from mxnet import gluon

from .utils import rcnn_split_and_load, get_network
from autogluon.core.task.base.base_predictor import BasePredictor
from autogluon.core import AutoGluonObject
from autogluon.core.utils import save, load
from autogluon.mxnet.utils import collect_params, update_params

__all__ = ['Detector']


class Detector(BasePredictor):
    """
    Trained Object Detector returned by `task.fit()`
    """

    def __init__(self, model, results, scheduler_checkpoint, args, format_results=True, **kwargs):
        self.model = model
        self.results = self._format_results(results) if format_results else results
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
        net.collect_params().reset_ctx(ctx)

        def _get_dataloader(net, test_dataset, data_shape, batch_size, num_workers, num_devices,
                            args):
            """Get dataloader."""
            if args.meta_arch == 'yolo3':
                width, height = data_shape, data_shape
                val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
                test_loader = gluon.data.DataLoader(
                    test_dataset.transform(YOLO3DefaultValTransform(width, height)),
                    batch_size,
                    False,
                    batchify_fn=val_batchify_fn,
                    last_batch='keep',
                    num_workers=num_workers
                )
                return test_loader
            elif args.meta_arch == 'faster_rcnn':
                """Get faster rcnn dataloader."""
                test_bfn = Tuple(*[Append() for _ in range(3)])
                short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
                # validation use 1 sample per device
                test_loader = gluon.data.DataLoader(
                    test_dataset.transform(FasterRCNNDefaultValTransform(short, net.max_size)),
                    num_devices,
                    False,
                    batchify_fn=test_bfn,
                    last_batch='keep',
                    num_workers=args.num_workers
                )
                return test_loader
            else:
                raise NotImplementedError('%s not implemented.' % args.meta_arch)

        def _validate(net, val_data, ctx, eval_metric):
            """Test on validation dataset."""
            eval_metric.reset()
            # set nms threshold and topk constraint
            if args.meta_arch == 'yolo3':
                net.set_nms(nms_thresh=0.45, nms_topk=400)
            mx.nd.waitall()
            net.hybridize()
            for batch in val_data:
                if args.meta_arch == 'yolo3':
                    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0,
                                                      even_split=False)
                    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0,
                                                       even_split=False)
                    split_batch = data, label
                elif args.meta_arch == 'faster_rcnn':
                    split_batch = rcnn_split_and_load(batch, ctx_list=ctx)
                    clipper = gcv.nn.bbox.BBoxClipToImage()
                else:
                    raise NotImplementedError('%s not implemented.' % args.meta_arch)
                det_bboxes = []
                det_ids = []
                det_scores = []
                gt_bboxes = []
                gt_ids = []
                gt_difficults = []
                for data in zip(*split_batch):
                    if args.meta_arch == 'yolo3':
                        x, y = data
                    elif args.meta_arch == 'faster_rcnn':
                        x, y, im_scale = data
                    else:
                        raise NotImplementedError('%s not implemented.' % args.meta_arch)
                    # get prediction results
                    ids, scores, bboxes = net(x)
                    det_ids.append(ids)
                    det_scores.append(scores)
                    # clip to image size
                    if args.meta_arch == 'yolo3':
                        det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                    elif args.meta_arch == 'faster_rcnn':
                        det_bboxes.append(clipper(bboxes, x))
                        # rescale to original resolution
                        im_scale = im_scale.reshape((-1)).asscalar()
                        det_bboxes[-1] *= im_scale
                    else:
                        raise NotImplementedError('%s not implemented.' % args.meta_arch)
                    # split ground truths
                    gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                    gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                    if args.meta_arch == 'faster_rcnn':
                        gt_bboxes[-1] *= im_scale
                    gt_difficults.append(
                        y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
                # update metric
                if args.meta_arch == 'yolo3':
                    eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids,
                                       gt_difficults)
                elif args.meta_arch == 'faster_rcnn':
                    for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in \
                            zip(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults):
                        eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
                else:
                    raise NotImplementedError('%s not implemented.' % args.meta_arch)
            return eval_metric.get()

        if isinstance(dataset, AutoGluonObject):
            dataset = dataset.init()
        test_dataset, eval_metric = dataset.get_dataset_and_metric()
        test_data = _get_dataloader(net, test_dataset, args.data_shape, args.batch_size,
                                    args.num_workers, len(ctx), args)
        return _validate(net, test_data, ctx, eval_metric)

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
        net.collect_params().reset_ctx(ctx=mx.cpu())

        x, img = presets.yolo.load_test(X, short=512)
        ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]

        if plot:
            gcv.utils.viz.plot_bbox(img, bboxes, scores, ids,
                                    thresh=thresh,
                                    class_names=net.classes,
                                    ax=None)
        plt.show()
        return ids, scores, bboxes

    @classmethod
    def load(cls, checkpoint):
        """ load trained object detector from the file specified by 'checkpoint'
        """
        state_dict = load(checkpoint)
        args = state_dict['args']
        results = pkl.loads(state_dict['results'])
        scheduler_checkpoint = state_dict['scheduler_checkpoint']
        model_params = state_dict['model_params']
        classes = state_dict['classes']

        model = get_network(args.meta_arch, args.net, classes, ctx=mx.cpu(0), syncbn=args.syncbn)
        update_params(model, model_params)

        return cls(model, results, scheduler_checkpoint, args, format_results=False)

    def state_dict(self, destination=None):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        model_params = collect_params(self.model)
        destination['model_params'] = model_params
        destination['results'] = pkl.dumps(self.results)
        destination['scheduler_checkpoint'] = self.scheduler_checkpoint
        destination['args'] = self.args
        destination['classes'] = destination['args'].pop('dataset').get_classes()
        return destination

    def save(self, checkpoint):
        """save object detector to the file specified by 'checkpoint'
        """
        state_dict = self.state_dict()
        save(state_dict, checkpoint)

    def evaluate_predictions(self, y_true, y_pred):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError
