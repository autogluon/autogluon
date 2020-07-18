import numpy as np
import os
import math
import logging
import time
import json
import mxnet as mx
import uuid
from mxnet.util import use_np
from mxnet.lr_scheduler import PolyScheduler, CosineScheduler
from mxnet.gluon.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from scipy.stats import pearsonr, spearmanr
from ....contrib.nlp.models import get_backbone
from ....contrib.nlp.lr_scheduler import InverseSquareRootScheduler
from ....contrib.nlp.utils.config import CfgNode
from ....contrib.nlp.utils.misc import set_seed, logging_config, parse_ctx, grouper,\
    count_parameters, repeat, get_mxnet_visible_gpus
from ....contrib.nlp.utils.parameter import move_to_ctx, clip_grad_global_norm
from ....contrib.nlp.utils.registry import Registry
from .. import constants as _C
from ....core import args
from ....scheduler import FIFOScheduler
from ..column_property import get_column_property_metadata, get_column_properties_from_metadata
from ..preprocessing import TabularBasicBERTPreprocessor
from ..modules.basic_prediction import BERTForTabularBasicV1
from ..dataset import TabularDataset, infer_problem_type


@use_np
def get_optimizer(cfg, updates_per_epoch):
    max_update = int(updates_per_epoch * cfg.num_train_epochs)
    warmup_steps = int(updates_per_epoch * cfg.num_train_epochs * cfg.warmup_portion)
    if cfg.lr_scheduler == 'triangular':
        assert warmup_steps < max_update
        lr_scheduler = PolyScheduler(max_update=max_update,
                                     base_lr=cfg.lr,
                                     warmup_begin_lr=cfg.begin_lr,
                                     pwr=1,
                                     final_lr=cfg.final_lr,
                                     warmup_steps=warmup_steps,
                                     warmup_mode='linear')
    elif cfg.lr_scheduler == 'inv_sqrt':
        warmup_steps = int(updates_per_epoch * cfg.num_train_epochs
                           * cfg.warmup_portion)
        lr_scheduler = InverseSquareRootScheduler(warmup_steps=warmup_steps,
                                                  base_lr=cfg.lr,
                                                  warmup_init_lr=cfg.begin_lr)
    elif cfg.lr_scheduler == 'constant':
        lr_scheduler = None
    elif cfg.lr_scheduler == 'cosine':
        max_update = int(updates_per_epoch * cfg.num_train_epochs)
        warmup_steps = int(updates_per_epoch * cfg.num_train_epochs
                           * cfg.warmup_portion)
        assert warmup_steps < max_update
        lr_scheduler = CosineScheduler(max_update=max_update,
                                       base_lr=cfg.lr,
                                       final_lr=cfg.final_lr,
                                       warmup_steps=warmup_steps,
                                       warmup_begin_lr=cfg.begin_lr)
    else:
        raise ValueError('Unsupported lr_scheduler="{}"'
                         .format(cfg.lr_scheduler))
    optimizer_params = {'learning_rate': cfg.lr,
                        'wd': cfg.wd,
                        'lr_scheduler': lr_scheduler}
    optimizer = cfg.optimizer
    additional_params = {key: value for key, value in cfg.optimizer_params}
    optimizer_params.update(additional_params)
    return optimizer, optimizer_params, max_update


@use_np
def apply_layerwise_decay(model, layerwise_decay, backbone_name, not_included=None):
    """Apply the layer-wise gradient decay
    .. math::
        lr = lr * layerwise_decay^(max_depth - layer_depth)

    Parameters:
    ----------
    model
        qa_net
    layerwise_decay: int
        layer-wise decay power
    not_included: list of str
        A list or parameter names that not included in the layer-wise decay
    """
    if not_included is None:
        not_included = []
    # consider the task specific fine-tuning layer as the last layer, following with pooler
    # In addition, the embedding parameters have the smaller learning rate based on this setting.
    if 'electra' in backbone_name:
        all_layers = model.encoder.all_encoder_layers
    else:
        all_layers = model.encoder.all_layers
    max_depth = len(all_layers) + 2
    for key, value in model.collect_params().items():
        if 'scores' in key:
            value.lr_mult = layerwise_decay ** 0
        if 'pool' in key:
            value.lr_mult = layerwise_decay ** 1
        if 'embed' in key:
            value.lr_mult = layerwise_decay ** max_depth

    for (layer_depth, layer) in enumerate(all_layers):
        layer_params = layer.collect_params()
        for key, value in layer_params.items():
            for pn in not_included:
                if pn in key:
                    continue
            value.lr_mult = layerwise_decay**(max_depth - (layer_depth + 1))


def base_optimization_config():
    """The basic optimization phase"""
    cfg = CfgNode()
    cfg.lr_scheduler = 'triangular'
    cfg.optimizer = 'adamw'
    cfg.optimizer_params = [('beta1', 0.9),
                            ('beta2', 0.999),
                            ('epsilon', 1e-6),
                            ('correct_bias', False)]
    cfg.begin_lr = 0.0
    cfg.batch_size = 32
    cfg.model_average = 5
    cfg.per_device_batch_size = 16  # Per-device batch-size
    cfg.val_batch_size_mult = 2  # By default, we double the batch size for validation
    cfg.lr = 1E-4
    cfg.final_lr = 0.0
    cfg.num_train_epochs = 3
    cfg.warmup_portion = 0.1
    cfg.layerwise_lr_decay = 0.8  # The layer_wise decay
    cfg.wd = 0.01  # Weight Decay
    cfg.max_grad_norm = 1.0  # Maximum Gradient Norm
    # The validation frequency = validation frequency * num_updates_in_an_epoch
    cfg.valid_frequency = 0.2
    # Logging frequency = log frequency * num_updates_in_an_epoch
    cfg.log_frequency = 0.1
    return cfg


def base_model_config():
    cfg = CfgNode()
    cfg.preprocess = CfgNode()
    cfg.preprocess.merge_text = True
    cfg.preprocess.max_length = 128
    cfg.backbone = CfgNode()
    cfg.backbone.name = 'google_electra_base'
    cfg.network = BERTForTabularBasicV1.get_cfg()
    return cfg


def base_learning_config():
    cfg = CfgNode()
    cfg.early_stopping_patience = 5  # Stop if we cannot find better checkpoints
    cfg.valid_ratio = 0.15      # The ratio of dataset to split for validation
    cfg.stop_metric = 'auto'    # Automatically define the stopping metric
    cfg.log_metrics = 'auto'    # Automatically determine the metrics used in logging
    return cfg


def base_misc_config():
    cfg = CfgNode()
    cfg.seed = 123
    cfg.context = 'gpu0'
    cfg.exp_dir = './autonlp'
    return cfg


def base_cfg():
    cfg = CfgNode()
    cfg.version = 1
    cfg.optimization = base_optimization_config()
    cfg.learning = base_learning_config()
    cfg.model = base_model_config()
    cfg.misc = base_misc_config()
    cfg.freeze()
    return cfg


def electra_base():
    """The search space of Electra Base"""
    cfg = base_cfg()
    cfg.defrost()
    cfg.optimization.layerwise_lr_decay = 0.8
    cfg.freeze()
    return cfg


def mobile_bert():
    """The search space of MobileBERT"""
    cfg = base_cfg()
    cfg.defrost()
    cfg.optimization.layerwise_lr_decay = -1
    cfg.model.backbone.name = 'google_uncased_mobilebert'
    cfg.optimization.lr = 1E-5
    cfg.optimization.num_train_epochs = 5.0
    cfg.freeze()
    return cfg


def calculate_metric_scores(metrics, predictions, gt_labels,
                            pos_label=1):
    """

    Parameters
    ----------
    metrics
        A list of metric names
    predictions
    gt_labels
    pos_label

    Returns
    -------
    metric_scores
        A dictionary contains key --> metric scores
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    metric_scores = dict()
    for metric_name in metrics:
        if metric_name == 'acc':
            metric_scores[metric_name] = accuracy_score(gt_labels,
                                                        predictions.argmax(axis=-1))
        elif metric_name == 'f1':
            metric_scores[metric_name] = f1_score(gt_labels,
                                                  predictions.argmax(axis=-1),
                                                  pos_label=pos_label)
        elif metric_name == 'mcc':
            metric_scores[metric_name] = matthews_corrcoef(gt_labels,
                                                           predictions.argmax(axis=-1))
        elif metric_name == 'auc':
            metric_scores[metric_name] = roc_auc_score(gt_labels,
                                                       predictions[:, pos_label])
        elif metric_name == 'nll':
            metric_scores[metric_name]\
                = - np.log(predictions[np.arange(gt_labels.shape[0]),
                                       gt_labels]).mean()
        elif metric_name == 'pearsonr':
            metric_scores[metric_name] = pearsonr(gt_labels, predictions)[0]
        elif metric_name == 'spearmanr':
            metric_scores[metric_name] = spearmanr(gt_labels, predictions)[0]
        elif metric_name == 'mse':
            metric_scores[metric_name] = np.square(predictions - gt_labels).mean()
        elif metric_name == 'rmse':
            metric_scores[metric_name] = np.sqrt(np.square(predictions - gt_labels).mean())
        elif metric_name == 'mae':
            metric_scores[metric_name] = np.abs(predictions - gt_labels).mean()
        else:
            raise ValueError('Unknown metric = {}'.format(metric_name))
    return metric_scores


def is_better_score(metric_name, baseline, new_score):
    """Whether the new score is better than the baseline

    Parameters
    ----------
    metric_name
        Name of the metric
    baseline
        The baseline score
    new_score
        The new score

    Returns
    -------
    ret
        Whether the new score is better than the baseline
    """
    if metric_name in ['acc', 'f1', 'mcc', 'auc', 'pearsonr', 'spearmanr']:
        return new_score > baseline
    elif metric_name in ['mse', 'rmse', 'mae']:
        return new_score < baseline
    else:
        raise NotImplementedError


@use_np
def _classification_regression_predict(net, dataloader,
                                       problem_type, ctx_l, has_label=True):
    """

    Parameters
    ----------
    net
        The network
    dataloader
        The dataloader
    problem_type
        Types of the labels
    ctx_l

    has_label


    Returns
    -------
    predictions
    """
    predictions = []
    for sample_l in grouper(dataloader, len(ctx_l)):
        iter_pred_l = []
        for sample, ctx in zip(sample_l, ctx_l):
            if sample is None:
                continue
            if has_label:
                batch_feature, batch_label = sample
            else:
                batch_feature = sample
            batch_feature = move_to_ctx(batch_feature, ctx)
            pred = net(batch_feature)
            if problem_type == _C.CLASSIFICATION:
                pred = mx.npx.softmax(pred, axis=-1)
            iter_pred_l.append(pred)
        for pred in iter_pred_l:
            predictions.append(pred.asnumpy())
    predictions = np.concatenate(predictions, axis=0)
    return predictions


@use_np
class BertForTextPredictionBasic:
    def __init__(self, column_properties, label_columns, feature_columns,
                 label_shapes, problem_types, eval_metric,
                 stopping_metric, log_metrics, output_directory, logger,
                 base_config=None, search_space=None):
        """

        Parameters
        ----------
        column_properties
        label_columns
        feature_columns
        label_shapes
        problem_types
        eval_metric
        stopping_metric
        log_metrics
        output_directory
        logger
        base_config
        search_space
        """
        super(BertForTextPredictionBasic, self).__init__()
        if base_config is None:
            self._base_config = base_cfg()
        else:
            self._base_config = base_cfg().clone_merge(base_config)
        self._base_config.defrost()
        if output_directory is not None:
            self._base_config.misc.exp_dir = output_directory
        self._base_config.freeze()
        self._search_space = search_space
        self._column_properties = column_properties
        self._stopping_metric = stopping_metric
        self._eval_metric = eval_metric
        self._log_metrics = log_metrics
        self._logger = logger
        self._output_directory = output_directory

        self._label_columns = label_columns
        self._feature_columns = feature_columns
        self._label_shapes = label_shapes
        self._problem_types = problem_types

        # Need to be set in the fit call
        self._net = None
        self._preprocessor = None
        self._train_data = None
        self._tuning_data = None
        self._config = None

    @property
    def search_space(self):
        return self._search_space

    @property
    def base_config(self):
        return self._base_config

    @property
    def problem_types(self):
        return self._problem_types

    @staticmethod
    def default_config():
        """Get the default configuration

        Returns
        -------
        cfg
            The configuration specified by the key
        """
        return base_cfg()

    def _train_function(self, args=None, reporter=None):
        start_tick = time.time()
        cfg = self.base_config.clone()
        specified_values = []
        for key in self._search_space:
            specified_values.append(key)
            # TODO(?) Fix this!
            #  We need to replace here due to issue: https://github.com/awslabs/autogluon/issues/560
            specified_values.append(getattr(args, key.replace('.', '___')))
        cfg.merge_from_list(specified_values)
        exp_dir = cfg.misc.exp_dir
        if reporter is not None:
            task_id = args.task_id
            printable_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
            exp_dir = os.path.join(exp_dir, 'task{}_{}'.format(task_id, printable_time))
            os.makedirs(exp_dir)
            cfg.defrost()
            cfg.misc.exp_dir = exp_dir
            cfg.freeze()
        logging_config(folder=exp_dir, name='training')
        logging.info(cfg)
        # Load backbone model
        backbone_model_cls, backbone_cfg, tokenizer, backbone_params_path, _ \
            = get_backbone(cfg.model.backbone.name)
        with open(os.path.join(exp_dir, 'cfg.yml'), 'w') as f:
            f.write(str(cfg))
        with open(os.path.join(exp_dir, 'backbone_cfg.yml'), 'w') as f:
            f.write(str(backbone_cfg))
        text_backbone = backbone_model_cls.from_cfg(backbone_cfg)
        # Build Preprocessor + Preprocess the training dataset + Inference problem type
        # TODO Move preprocessor + Dataloader to outer loop to better cache the dataloader
        preprocessor = TabularBasicBERTPreprocessor(tokenizer=tokenizer,
                                                    column_properties=self._column_properties,
                                                    label_columns=self._label_columns,
                                                    max_length=cfg.model.preprocess.max_length,
                                                    merge_text=cfg.model.preprocess.merge_text)
        self._preprocessor = preprocessor
        logging.info('Process training set...')
        processed_train = preprocessor.process_train(self._train_data.table)
        logging.info('Done!')
        logging.info('Process dev set...')
        processed_dev = preprocessor.process_test(self._tuning_data.table)
        logging.info('Done!')
        label = self._label_columns[0]
        # Get the ground-truth dev labels
        gt_dev_labels = np.array(self._tuning_data.table[label].apply(
            self._column_properties[label].transform))
        gpu_ctx_l = get_mxnet_visible_gpus()
        if len(gpu_ctx_l) == 0:
            ctx_l = [mx.cpu()]
        else:
            ctx_l = gpu_ctx_l
        base_batch_size = cfg.optimization.per_device_batch_size
        num_accumulated = int(np.ceil(cfg.optimization.batch_size / base_batch_size))
        inference_base_batch_size = base_batch_size * cfg.optimization.val_batch_size_mult
        train_dataloader = DataLoader(processed_train,
                                      batch_size=base_batch_size,
                                      shuffle=True,
                                      batchify_fn=preprocessor.batchify(is_test=False))
        dev_dataloader = DataLoader(processed_dev,
                                    batch_size=inference_base_batch_size,
                                    shuffle=False,
                                    batchify_fn=preprocessor.batchify(is_test=True))
        net = BERTForTabularBasicV1(text_backbone=text_backbone,
                                    feature_field_info=preprocessor.feature_field_info(),
                                    label_shape=self._label_shapes[0],
                                    cfg=cfg.model.network)
        net.initialize_with_pretrained_backbone(backbone_params_path, ctx=ctx_l)
        net.hybridize()
        num_total_params, num_total_fixed_params = count_parameters(net.collect_params())
        logging.info('#Total Params/Fixed Params={}/{}'.format(num_total_params,
                                                               num_total_fixed_params))
        # Initialize the optimizer
        updates_per_epoch = int(len(train_dataloader) / (num_accumulated * len(ctx_l)))
        optimizer, optimizer_params, max_update\
            = get_optimizer(cfg.optimization,
                            updates_per_epoch=updates_per_epoch)
        valid_interval = math.ceil(cfg.optimization.valid_frequency * updates_per_epoch)
        train_log_interval = math.ceil(cfg.optimization.log_frequency * updates_per_epoch)
        trainer = mx.gluon.Trainer(net.collect_params(),
                                   optimizer, optimizer_params,
                                   update_on_kvstore=False)
        if cfg.optimization.layerwise_lr_decay > 0:
            apply_layerwise_decay(net.text_backbone,
                                  cfg.optimization.layerwise_lr_decay,
                                  backbone_name=cfg.model.backbone.name)
        # Do not apply weight decay to all the LayerNorm and bias
        for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0
        params = [p for p in net.collect_params().values() if p.grad_req != 'null']

        # Set grad_req if gradient accumulation is required
        if num_accumulated > 1:
            logging.info('Using gradient accumulation. Global batch size = {}'
                         .format(cfg.optimization.batch_size))
            for p in params:
                p.grad_req = 'add'
            net.collect_params().zero_grad()
        train_loop_dataloader = grouper(repeat(train_dataloader), len(ctx_l))
        log_loss_l = [mx.np.array(0.0, dtype=np.float32, ctx=ctx) for ctx in ctx_l]
        log_num_samples_l = [0 for _ in ctx_l]
        logging_start_tick = time.time()
        best_dev_metric = None
        mx.npx.waitall()
        no_better_rounds = 0
        for update_idx in range(max_update):
            num_samples_per_update_l = [0 for _ in ctx_l]
            for accum_idx in range(num_accumulated):
                sample_l = next(train_loop_dataloader)
                loss_l = []
                num_samples_l = [0 for _ in ctx_l]
                for i, (sample, ctx) in enumerate(zip(sample_l, ctx_l)):
                    feature_batch, label_batch = sample
                    feature_batch = move_to_ctx(feature_batch, ctx)
                    label_batch = move_to_ctx(label_batch, ctx)
                    with mx.autograd.record():
                        pred = net(feature_batch)
                        if self._problem_types[0] == _C.CLASSIFICATION:
                            logits = mx.npx.log_softmax(pred, axis=-1)
                            loss = - mx.npx.pick(logits, label_batch[0])
                        elif self._problem_types[0] == _C.REGRESSION:
                            loss = mx.np.square(pred - label_batch[0])
                        loss_l.append(loss.mean() / len(ctx_l))
                        num_samples_l[i] = loss.shape[0]
                        num_samples_per_update_l[i] += loss.shape[0]
                for loss in loss_l:
                    loss.backward()
                for i in range(len(ctx_l)):
                    log_loss_l[i] += loss_l[i] * len(ctx_l) * num_samples_l[i]
                    log_num_samples_l[i] += num_samples_per_update_l[i]
            # Begin to update
            trainer.allreduce_grads()
            num_samples_per_update = sum(num_samples_per_update_l)
            total_norm, ratio, is_finite = \
                clip_grad_global_norm(params, cfg.optimization.max_grad_norm * num_accumulated)
            total_norm = total_norm / num_accumulated
            trainer.update(num_samples_per_update)

            # Clear after update
            if num_accumulated > 1:
                net.collect_params().zero_grad()
            if (update_idx + 1) % train_log_interval == 0:
                log_loss = sum([ele.as_in_ctx(ctx_l[0]) for ele in log_loss_l]).asnumpy()
                log_num_samples = sum(log_num_samples_l)
                logging.info(
                    '[Iter {}/{}, Epoch {}] train loss={}, gnorm={}, lr={}, #samples processed={},'
                    ' #sample per second={}'
                    .format(update_idx + 1, max_update, int(update_idx / updates_per_epoch),
                            log_loss / log_num_samples, total_norm, trainer.learning_rate,
                            log_num_samples,
                            log_num_samples / (time.time() - logging_start_tick)))
                logging_start_tick = time.time()
                log_loss_l = [mx.np.array(0.0, dtype=np.float32, ctx=ctx) for ctx in ctx_l]
                log_num_samples_l = [0 for _ in ctx_l]
            if (update_idx + 1) % valid_interval == 0 or (update_idx + 1) == max_update:
                valid_start_tick = time.time()
                dev_predictions =\
                    _classification_regression_predict(net, dataloader=dev_dataloader,
                                                       ctx_l=ctx_l,
                                                       problem_type=self._problem_types[0],
                                                       has_label=False)
                metric_scores = calculate_metric_scores(self._log_metrics,
                                                        predictions=dev_predictions,
                                                        gt_labels=gt_dev_labels)
                valid_time_spent = time.time() - valid_start_tick
                if best_dev_metric is None or is_better_score(self._stopping_metric,
                                                              best_dev_metric,
                                                              metric_scores[self._stopping_metric]):
                    find_better = True
                    no_better_rounds = 0
                    best_dev_metric = metric_scores[self._stopping_metric]
                else:
                    find_better = False
                    no_better_rounds += 1
                if find_better:
                    net.save_parameters(os.path.join(exp_dir, 'best_model.params'))
                loss_string = ', '.join(['{}={}'.format(key, metric_scores[key])
                                         for key in self._log_metrics])
                logging.info('[Iter {}/{}, Epoch {}] valid {}, time spent={}'.format(
                    update_idx + 1, max_update, int(update_idx / updates_per_epoch),
                    loss_string, valid_time_spent))
                report_items = [('iteration', update_idx + 1),
                                ('epoch', int(update_idx / updates_per_epoch))] + \
                               [(k, v.item()) for k, v in metric_scores.items()] + \
                               [('fine_better', find_better),
                                ('time_spent', time.time() - start_tick)]
                if isinstance(self._eval_metric, str):
                    eval_metric_score = metric_scores[self._eval_metric].item()
                else:
                    eval_metric_score = np.mean([metric_scores[ele]
                                                 for ele in self._eval_metric]).item()
                report_items.append(('eval_metric_score', eval_metric_score))
                reporter(**dict(report_items))
                if no_better_rounds >= cfg.LEARNING.early_stopping_patience:
                    logging.info('Early stopping patience reached!')
                    break
        net.load_parameters(filename=os.path.join(exp_dir, 'best_model.params'))
        return net

    def train(self, train_data, tuning_data, label_columns, feature_columns,
              resources, time_limits=None):
        self._column_properties = train_data.column_properties
        self._label_columns = label_columns
        self._feature_columns = feature_columns
        assert len(self._label_columns) == 1
        self._train_data = train_data
        self._tuning_data = tuning_data
        os.makedirs(self._output_directory, exist_ok=True)
        # TODO(?) Fix this!
        #  We need to replace here due to issue: https://github.com/awslabs/autogluon/issues/560
        search_space_decorator = args(**{key.replace('.', '___'): value
                                         for key, value in self.search_space.items()})
        train_fn = search_space_decorator(self._train_function)
        scheduler = FIFOScheduler(train_fn,
                                  time_out=time_limits,
                                  num_trials=10,
                                  resource=resources,
                                  checkpoint=os.path.join(self._output_directory,
                                                          'scheduler.checkpoint'),
                                  reward_attr='eval_metric_score',
                                  time_attr='time_spent')
        scheduler.run()
        scheduler.join_jobs(timeout=time_limits)

    def evaluate(self, valid_data, metrics):
        assert self.net is not None
        if not isinstance(valid_data, TabularDataset):
            valid_data = TabularDataset(valid_data,
                                        column_properties=self._column_properties)
        ground_truth = np.array(valid_data.table[self._label].apply(
            self._column_properties[self._label].transform))
        if self._problem_types[0] == _C.CLASSIFICATION:
            predictions = self.predict_proba(valid_data)
        else:
            predictions = self.predict(valid_data)
        metric_scores = calculate_metric_scores(metrics=metrics,
                                                predictions=predictions,
                                                gt_labels=ground_truth)
        return metric_scores

    def _internal_predict(self, test_data, get_original_labels=True, get_probabilities=False):
        assert self.net is not None
        cfg = self.config
        if not isinstance(test_data, TabularDataset):
            test_data = TabularDataset(test_data,
                                       column_properties=self._column_properties)
        backbone_model_cls, backbone_cfg, tokenizer, backbone_params_path, _ \
            = get_backbone(cfg.model.backbone.name)
        processed_test = self._preprocessor.process_test(test_data.table)
        ctx_l = parse_ctx(cfg.MISC.context)
        base_batch_size = cfg.optimization.per_device_batch_size
        inference_batch_size = base_batch_size * cfg.optimization.val_batch_size_mult
        test_dataloader = DataLoader(processed_test,
                                     batch_size=inference_batch_size,
                                     shuffle=False,
                                     batchify_fn=self._preprocessor.batchify(is_test=True))
        test_predictions = _classification_regression_predict(self._net,
                                                              dataloader=test_dataloader,
                                                              ctx_l=ctx_l,
                                                              problem_type=self._problem_types[0],
                                                              has_label=False)
        if self._problem_types[0] == _C.CLASSIFICATION:
            if get_probabilities:
                return test_predictions
            else:
                test_predictions = test_predictions.argmax(axis=-1)
                if get_original_labels:
                    test_predictions = np.array(
                        list(map(self._column_properties[self._label].inv_transform,
                                 test_predictions)))
        return test_predictions

    def predict_proba(self, test_data):
        """

        Parameters
        ----------
        test_data
            The test data

        Returns
        -------
        probabilities
            The probabilities. Shape (#Samples, num_class)
        """
        assert self.problem_types[0] == _C.CLASSIFICATION
        return self._internal_predict(test_data,
                                      get_original_labels=False,
                                      get_probabilities=True)

    def predict(self, test_data, get_original_labels=True):
        """

        Parameters
        ----------
        test_data
            tabular dataset
        get_original_labels
            Whether to get the original labels

        Returns
        -------
        predictions
            The predictions. Shape (#Samples,)
        """
        return self._internal_predict(test_data,
                                      get_original_labels=get_original_labels,
                                      get_probabilities=False)

    def save(self, dir_path):
        """Save the trained model to a directory

        Parameters
        ----------
        dir_path
            The destination directory
        """
        os.makedirs(dir_path, exist_ok=True)
        self.net.save_parameters(os.path.join(dir_path, 'net.params'))
        with open(os.path.join(dir_path, 'cfg.yml'), 'w') as of:
            of.write(self.config.dump())
        with open(os.path.join(dir_path, 'column_metadata.json'), 'w') as of:
            json.dump(get_column_property_metadata(self._column_properties),
                      of, ensure_ascii=True)
        with open(os.path.join(dir_path, 'assets.json'), 'w') as of:
            json.dump({'label': self._label,
                       'label_shape': self._label_shape,
                       'problem_types': self._problem_types,
                       'feature_columns': self._feature_columns},
                      of, ensure_ascii=True)

    @classmethod
    def load(cls, dir_path):
        """Load the trained model from a directory

        Parameters
        ----------
        dir_path
            The directory path

        Returns
        -------
        model
        """
        loaded_config = cls.get_cfg().clone_merge(os.path.join(dir_path, 'cfg.yml'))
        model = cls(loaded_config)
        with open(os.path.join(dir_path, 'assets.json'), 'r') as f:
            assets = json.load(f)
        model._label = assets['label']
        model._label_shape = assets['label_shape']
        model._problem_type = assets['problem_type']
        model._feature_columns = assets['feature_columns']
        backbone_model_cls, backbone_cfg, tokenizer, backbone_params_path, _ \
            = get_backbone(loaded_config.model.backbone.name)
        model._column_properties = get_column_properties_from_metadata(
            os.path.join(dir_path, 'column_metadata.json'))
        # Initialize the preprocessor
        preprocessor = TabularBasicBERTPreprocessor(
            tokenizer=tokenizer,
            column_properties=model.column_properties,
            label_columns=model._label,
            max_length=loaded_config.model.preprocess.max_length,
            merge_text=loaded_config.model.preprocess.merge_text)
        text_backbone = backbone_model_cls.from_cfg(backbone_cfg)
        model._preprocessor = preprocessor
        model._net = BERTForTabularBasicV1(text_backbone=text_backbone,
                                           feature_field_info=preprocessor.feature_field_info(),
                                           label_shape=model.label_shape,
                                           cfg=loaded_config.model.network)
        return model
