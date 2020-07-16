import numpy as np
import os
import math
import logging
import time
import json
import mxnet as mx
import autogluon as ag
from mxnet.util import use_np
from mxnet.lr_scheduler import PolyScheduler, CosineScheduler
from mxnet.gluon.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from scipy.stats import pearsonr, spearmanr
from .. import constants as _C
from ...models import get_backbone
from ..column_property import get_column_property_metadata, get_column_properties_from_metadata
from ..preprocessing import TabularBasicBERTPreprocessor
from ...lr_scheduler import InverseSquareRootScheduler
from ..modules.classification import BERTForTabularBasicV1
from ...utils.config import CfgNode
from ...utils.misc import set_seed, logging_config, parse_ctx, grouper, count_parameters, repeat
from ...utils.parameter import move_to_ctx, clip_grad_global_norm
from ...utils.registry import Registry
from .base import BaseEstimator
from ..dataset import TabularDataset, random_split_train_val

v1_prebuild_search_space = Registry('v1_prebuild_search_space')


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
def apply_layerwise_decay(model, layerwise_decay, not_included=None):
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
    all_layers = model.encoder.all_encoder_layers
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
    cfg.num_accumulated = 2
    cfg.val_batch_size_mult = 2  # By default, we double the batch size for validation
    cfg.lr = 1E-4
    cfg.final_lr = 0.0
    cfg.num_train_epochs = 3.0
    cfg.warmup_portion = 0.1
    cfg.layerwise_lr_decay = -1  # The layer_wise decay
    cfg.wd = 0.01  # Weight Decay
    cfg.max_grad_norm = 1.0  # Maximum Gradient Norm
    # The validation frequency = validation frequency * num_updates_in_an_epoch
    cfg.valid_frequency = 0.2
    # Logging frequency = log frequency * num_updates_in_an_epoch
    cfg.log_frequency = 0.1
    return cfg


def base_model_config():
    cfg = CfgNode()
    cfg.PREPROCESS = CfgNode()
    cfg.PREPROCESS.merge_text = True
    cfg.PREPROCESS.max_length = 128
    cfg.BACKBONE = CfgNode()
    cfg.BACKBONE.name = 'google_electra_base'
    cfg.NETWORK = BERTForTabularBasicV1.get_cfg()
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
    cfg.VERSION = 1
    cfg.OPTIMIZATION = base_optimization_config()
    cfg.LEARNING = base_learning_config()
    cfg.MODEL = base_model_config()
    cfg.MISC = base_misc_config()
    cfg.freeze()
    return cfg


@v1_prebuild_search_space.register()
def electra_base_fixed():
    """The search space of Electra Base"""
    cfg = base_cfg()
    cfg.defrost()
    cfg.OPTIMIZATION.layerwise_lr_decay = 0.8
    cfg.freeze()
    return cfg


@v1_prebuild_search_space.register()
def mobile_bert_fixed():
    """The search space of MobileBERT"""
    cfg = base_cfg()
    cfg.defrost()
    cfg.OPTIMIZATION.layerwise_lr_decay = -1
    cfg.BACKBONE.name = 'google_uncased_mobilebert'
    cfg.OPTIMIZATION.lr = 1E-5
    cfg.OPTIMIZATION.num_train_epochs = 5.0
    cfg.freeze()
    return cfg


def infer_stop_eval_metrics(problem_type, label_shape):
    """

    Parameters
    ----------
    problem_type
    label_shape

    Returns
    -------
    stop_metric
    log_metrics
    """
    if problem_type == _C.CLASSIFICATION:
        stop_metric = 'acc'
        if label_shape == 2:
            log_metrics = ['f1', 'mcc', 'auc', 'acc', 'nll']
        else:
            log_metrics = ['acc', 'nll']
    elif problem_type == _C.REGRESSION:
        stop_metric = 'mse'
        log_metrics = ['mse', 'rmse', 'mae']
    else:
        raise NotImplementedError
    return stop_metric, log_metrics


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
def _classification_regression_predict(net, dataloader, problem_type, ctx_l,
                                       has_label=True):
    """

    Parameters
    ----------
    net
    dataloader
    problem_type
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
class BertForTextPredictionBasic(BaseEstimator):
    def __init__(self, config=None, logger=None, reporter=None):
        super(BertForTextPredictionBasic, self).__init__(config=config,
                                                         logger=logger,
                                                         reporter=reporter)
        self._problem_type = None
        self._net = None
        self._preprocessor = None
        self._column_properties = None
        self._label = None
        self._label_shape = None
        self._feature_columns = None

    @property
    def problem_type(self):
        return self._problem_type

    @property
    def label_shape(self):
        return self._label_shape

    @property
    def label(self):
        return self._label

    @property
    def net(self):
        return self._net

    @staticmethod
    def get_cfg(key=None):
        """

        Parameters
        ----------
        key
            Prebuilt configurations

        Returns
        -------
        cfg
        """
        if key is None:
            return electra_base_fixed()
        else:
            return v1_prebuild_search_space.get(key)

    def fit(self, train_data, label, feature_columns=None, valid_data=None,
            time_limits=None):
        """Fit the train data with the given label

        Parameters
        ----------
        train_data
            The training data.
            Should be a format that can be converted to a tabular dataset
        label
            The label column
        feature_columns
            The feature columns
        valid_data
            The validation data
        time_limits
            The time limits in seconds
        """
        fit_start_tick = time.time()
        self._label = label
        cfg = self.config
        set_seed(cfg.MISC.seed)
        exp_dir = cfg.MISC.exp_dir
        logging_config(folder=exp_dir, name='train')
        ctx_l = parse_ctx(cfg.MISC.context)
        if feature_columns is None:
            feature_columns = [ele for ele in train_data.columns if ele != label]
        elif not isinstance(feature_columns, list):
            feature_columns = [feature_columns]
        self._feature_columns = feature_columns
        all_columns = feature_columns + [label]
        if not isinstance(train_data, TabularDataset):
            train_data = TabularDataset(train_data,
                                        columns=all_columns,
                                        label_columns=label)
        column_properties = train_data.column_properties
        self._column_properties = column_properties

        # Get the problem type + shape + metrics
        problem_type, label_shape = train_data.infer_problem_type(label_col_name=label)
        self._problem_type = problem_type
        self._label_shape = label_shape
        logging.info('Problem Type={}, Label Shape={}'.format(problem_type, label_shape))
        inferred_stop_metric, inferred_log_metrics = infer_stop_eval_metrics(self.problem_type,
                                                                             self.label_shape)
        if cfg.LEARNING.stop_metric == 'auto':
            stop_metric = inferred_stop_metric
        else:
            stop_metric = cfg.LEARNING.stop_metric
        if cfg.LEARNING.log_metrics == 'auto':
            log_metrics = inferred_log_metrics
        else:
            log_metrics = cfg.LEARNING.log_metrics.split(',')
        if stop_metric not in log_metrics:
            log_metrics.append(stop_metric)
        logging.info('Stop Metric={}, Log Metrics={}'.format(stop_metric, log_metrics))
        if valid_data is None:
            train_df, valid_df = random_split_train_val(train_data.table,
                                                        label=label,
                                                        valid_ratio=cfg.LEARNING.valid_ratio)
            train_data = TabularDataset(train_df, column_properties=column_properties)
            valid_data = TabularDataset(valid_df, column_properties=column_properties)
        else:
            if not isinstance(valid_data, TabularDataset):
                valid_data = TabularDataset(valid_data,
                                            label_columns=label,
                                            column_properties=column_properties)
        logging.info('Train Dataset:')
        logging.info(train_data)
        logging.info('Dev Dataset:')
        logging.info(valid_data)
        # Load backbone model
        backbone_model_cls, backbone_cfg, tokenizer, backbone_params_path, _ \
            = get_backbone(cfg.MODEL.BACKBONE.name)
        with open(os.path.join(exp_dir, 'cfg.yml'), 'w') as f:
            f.write(str(cfg))
        with open(os.path.join(exp_dir, 'backbone_cfg.yml'), 'w') as f:
            f.write(str(backbone_cfg))
        text_backbone = backbone_model_cls.from_cfg(backbone_cfg)
        # Build Preprocessor + Preprocess the training dataset + Inference problem type
        preprocessor = TabularBasicBERTPreprocessor(tokenizer=tokenizer,
                                                    column_properties=column_properties,
                                                    label_columns=label,
                                                    max_length=cfg.MODEL.PREPROCESS.max_length,
                                                    merge_text=cfg.MODEL.PREPROCESS.merge_text)
        self._preprocessor = preprocessor
        logging.info('Process training set...')
        processed_train = preprocessor.process_train(train_data.table)
        logging.info('Done!')
        logging.info('Process dev set...')
        processed_dev = preprocessor.process_test(valid_data.table)
        logging.info('Done!')
        # Get the ground-truth dev labels
        gt_dev_labels = np.array(valid_data.table[label].apply(column_properties[label].transform))
        np.save(os.path.join(exp_dir, 'gt_dev_labels.npy'), gt_dev_labels)
        batch_size = cfg.OPTIMIZATION.batch_size\
                     // len(ctx_l) // cfg.OPTIMIZATION.num_accumulated
        inference_batch_size = batch_size * cfg.OPTIMIZATION.val_batch_size_mult
        assert batch_size * cfg.OPTIMIZATION.num_accumulated * len(ctx_l)\
               == cfg.OPTIMIZATION.batch_size
        train_dataloader = DataLoader(processed_train,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      batchify_fn=preprocessor.batchify(is_test=False))
        dev_dataloader = DataLoader(processed_dev,
                                    batch_size=inference_batch_size,
                                    shuffle=False,
                                    batchify_fn=preprocessor.batchify(is_test=True))
        net = BERTForTabularBasicV1(text_backbone=text_backbone,
                                    feature_field_info=preprocessor.feature_field_info(),
                                    label_shape=label_shape,
                                    cfg=cfg.MODEL.NETWORK)
        self._net = net
        net.initialize_with_pretrained_backbone(backbone_params_path, ctx=ctx_l)
        net.hybridize()
        num_total_params, num_total_fixed_params = count_parameters(net.collect_params())
        logging.info('#Total Params/Fixed Params={}/{}'.format(num_total_params,
                                                               num_total_fixed_params))
        # Initialize the optimizer
        updates_per_epoch = int(
            len(train_dataloader) / (cfg.OPTIMIZATION.num_accumulated * len(ctx_l)))
        optimizer, optimizer_params, max_update = get_optimizer(cfg.OPTIMIZATION,
                                                                updates_per_epoch=updates_per_epoch)
        valid_interval = math.ceil(cfg.OPTIMIZATION.valid_frequency * updates_per_epoch)
        train_log_interval = math.ceil(cfg.OPTIMIZATION.log_frequency * updates_per_epoch)
        trainer = mx.gluon.Trainer(net.collect_params(),
                                   optimizer, optimizer_params,
                                   update_on_kvstore=False)
        if cfg.OPTIMIZATION.layerwise_lr_decay > 0:
            apply_layerwise_decay(net.text_backbone,
                                  cfg.OPTIMIZATION.layerwise_lr_decay)
        # Do not apply weight decay to all the LayerNorm and bias
        for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0
        params = [p for p in net.collect_params().values() if p.grad_req != 'null']

        # Set grad_req if gradient accumulation is required
        if cfg.OPTIMIZATION.num_accumulated > 1:
            logging.info('Using gradient accumulation. Global batch size = {}'
                         .format(cfg.OPTIMIZATION.batch_size))
            for p in params:
                p.grad_req = 'add'
            net.collect_params().zero_grad()
        train_loop_dataloader = grouper(repeat(train_dataloader), len(ctx_l))
        log_loss_l = [mx.np.array(0.0, dtype=np.float32, ctx=ctx) for ctx in ctx_l]
        log_num_samples_l = [0 for _ in ctx_l]
        logging_start_tick = time.time()
        best_dev_metric = None
        dev_metrics_csv_logger = open(os.path.join(exp_dir, 'metrics.csv'), 'w')
        dev_metrics_csv_logger.write(','.join(['update_idx', 'epoch']
                                              + log_metrics + ['find_better', 'time_spent']) + '\n')
        mx.npx.waitall()
        no_better_rounds = 0
        num_grad_accum = cfg.OPTIMIZATION.num_accumulated
        for update_idx in range(max_update):
            num_samples_per_update_l = [0 for _ in ctx_l]
            for accum_idx in range(num_grad_accum):
                sample_l = next(train_loop_dataloader)
                loss_l = []
                num_samples_l = [0 for _ in ctx_l]
                for i, (sample, ctx) in enumerate(zip(sample_l, ctx_l)):
                    feature_batch, label_batch = sample
                    feature_batch = move_to_ctx(feature_batch, ctx)
                    label_batch = move_to_ctx(label_batch, ctx)
                    with mx.autograd.record():
                        pred = net(feature_batch)
                        if problem_type == _C.CLASSIFICATION:
                            logits = mx.npx.log_softmax(pred, axis=-1)
                            loss = - mx.npx.pick(logits, label_batch[0])
                        elif problem_type == _C.REGRESSION:
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
                clip_grad_global_norm(params, cfg.OPTIMIZATION.max_grad_norm * num_grad_accum)
            total_norm = total_norm / num_grad_accum
            trainer.update(num_samples_per_update)

            # Clear after update
            if cfg.OPTIMIZATION.num_accumulated > 1:
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
                if time.time() - fit_start_tick > time_limits:
                    logging.info('Reached time limit. Stop!')
                    break
            if (update_idx + 1) % valid_interval == 0 or (update_idx + 1) == max_update:
                valid_start_tick = time.time()
                dev_predictions = _classification_regression_predict(net, dataloader=dev_dataloader,
                                                                     ctx_l=ctx_l,
                                                                     problem_type=problem_type,
                                                                     has_label=False)
                metric_scores = calculate_metric_scores(log_metrics,
                                                        predictions=dev_predictions,
                                                        gt_labels=gt_dev_labels)
                valid_time_spent = time.time() - valid_start_tick
                if best_dev_metric is None or is_better_score(stop_metric,
                                                              best_dev_metric,
                                                              metric_scores[stop_metric]):
                    find_better = True
                    no_better_rounds = 0
                    best_dev_metric = metric_scores[stop_metric]
                else:
                    find_better = False
                    no_better_rounds += 1
                if find_better:
                    net.save_parameters(os.path.join(exp_dir, 'best_model.params'))
                loss_string = ', '.join(['{}={}'.format(key, metric_scores[key])
                                         for key in log_metrics])
                logging.info('[Iter {}/{}, Epoch {}] valid {}, time spent={}'.format(
                    update_idx + 1, max_update, int(update_idx / updates_per_epoch),
                    loss_string, valid_time_spent))
                dev_metrics_csv_logger.write(','.join(
                    map(str, [update_idx + 1,
                              int(update_idx / updates_per_epoch)]
                        + [metric_scores[key] for key in log_metrics]
                        + [find_better, valid_time_spent])) + '\n')
                if no_better_rounds >= cfg.LEARNING.early_stopping_patience:
                    logging.info('Early stopping patience reached!')
                    break
        # TODO(sxjscience) Add SWA
        net.load_parameters(filename=os.path.join(exp_dir, 'best_model.params'))

    def evaluate(self, valid_data, metrics):
        assert self.net is not None
        if not isinstance(valid_data, TabularDataset):
            valid_data = TabularDataset(valid_data,
                                        column_properties=self._column_properties)
        ground_truth = np.array(valid_data.table[self._label].apply(
            self._column_properties[self._label].transform))
        if self.problem_type == _C.CLASSIFICATION:
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
            = get_backbone(cfg.MODEL.BACKBONE.name)
        processed_test = self._preprocessor.process_test(test_data.table)
        ctx_l = parse_ctx(cfg.MISC.context)
        batch_size = cfg.OPTIMIZATION.batch_size // len(ctx_l) // cfg.OPTIMIZATION.num_accumulated
        inference_batch_size = batch_size * cfg.OPTIMIZATION.val_batch_size_mult
        test_dataloader = DataLoader(processed_test,
                                     batch_size=inference_batch_size,
                                     shuffle=False,
                                     batchify_fn=self._preprocessor.batchify(is_test=True))
        test_predictions = _classification_regression_predict(self._net,
                                                              dataloader=test_dataloader,
                                                              ctx_l=ctx_l,
                                                              problem_type=self._problem_type,
                                                              has_label=False)
        if self.problem_type == _C.CLASSIFICATION:
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
        assert self.problem_type == _C.CLASSIFICATION
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
                       'problem_type': self._problem_type,
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
            = get_backbone(loaded_config.MODEL.BACKBONE.name)
        model._column_properties = get_column_properties_from_metadata(
            os.path.join(dir_path, 'column_metadata.json'))
        # Initialize the preprocessor
        preprocessor = TabularBasicBERTPreprocessor(
            tokenizer=tokenizer,
            column_properties=model.column_properties,
            label_columns=model._label,
            max_length=model.config.MODEL.PREPROCESS.max_length,
            merge_text=model.config.MODEL.PREPROCESS.merge_text)
        text_backbone = backbone_model_cls.from_cfg(backbone_cfg)
        model._preprocessor = preprocessor
        model._net = BERTForTabularBasicV1(text_backbone=text_backbone,
                                           feature_field_info=preprocessor.feature_field_info(),
                                           label_shape=model.label_shape,
                                           cfg=loaded_config.MODEL.NETWORK)
        return model
