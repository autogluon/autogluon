import numpy as np
import os
import math
import logging
import pandas as pd
import warnings
import time
import json
import pickle
import functools
import tqdm
from typing import Tuple
from autogluon.core.utils import set_logger_verbosity
from sklearn.preprocessing import LabelEncoder
import mxnet as mx
from mxnet.util import use_np
from mxnet.lr_scheduler import PolyScheduler, CosineScheduler
from mxnet.gluon.data import DataLoader
from autogluon_contrib_nlp.models import get_backbone
from autogluon_contrib_nlp.lr_scheduler import InverseSquareRootScheduler
from autogluon_contrib_nlp.utils.config import CfgNode
from autogluon_contrib_nlp.utils.misc import grouper, \
    count_parameters, repeat, get_mxnet_available_ctx
from autogluon_contrib_nlp.utils.parameter import move_to_ctx, clip_grad_global_norm

from autogluon.core import args, space
from autogluon.core.utils import in_ipynb, verbosity2loglevel
from autogluon.core.utils.utils import get_cpu_count, get_gpu_count
from autogluon.core.utils.loaders import load_pkl, load_pd
from autogluon.core.task.base import compile_scheduler_options_v2
from autogluon.core.task.base.base_task import schedulers
from autogluon.core.metrics import get_metric, Scorer
from autogluon.core.utils.multiprocessing_utils import force_forkserver
from autogluon.core.dataset import TabularDataset
from autogluon.core.decorator import sample_config
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.scheduler.reporter import FakeReporter

from .modules import MultiModalWithPretrainedTextNN
from .preprocessing import MultiModalTextFeatureProcessor, base_preprocess_cfg,\
    MultiModalTextBatchify, get_stats_string, auto_shrink_max_length, get_cls_sep_id
from .. import constants as _C
from ..utils import logging_config
from ..presets import ag_text_presets
from ... import version



@use_np
def get_optimizer(cfg, updates_per_epoch):
    """

    Parameters
    ----------
    cfg
        Configuration
    updates_per_epoch
        The number of updates per training epoch

    Returns
    -------
    optimizer
        The optimizer
    optimizer_params
        Optimization parameters
    max_update
        Maximum update
    """
    max_update = int(updates_per_epoch * cfg.num_train_epochs)
    warmup_steps = int(updates_per_epoch * cfg.num_train_epochs * cfg.warmup_portion)
    if cfg.lr_scheduler == 'triangular':
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
        The backbone model
    layerwise_decay: int
        layer-wise decay power
    not_included: list of str
        A list or parameter names that not included in the layer-wise decay
    """
    if not_included is None:
        not_included = []
    # consider the task specific fine-tuning layer as the last layer, following with pooler
    # In addition, the embedding parameters have the smaller learning rate based on this setting.
    if 'albert' in backbone_name:
        # Skip if it is the ALBERT model.
        return
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
            value.lr_mult = layerwise_decay ** (max_depth - (layer_depth + 1))


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
    cfg.nbest = 5                   # Keep the top K performed models
    cfg.per_device_batch_size = 16  # Per-device batch-size
    cfg.auto_per_device_batch_size = True  # Whether to automatically determine the runnable
                                           # per-device batch_size.
    cfg.val_batch_size_mult = 2  # By default, we double the batch size for validation
    cfg.lr = 1E-4
    cfg.final_lr = 0.0
    cfg.num_train_epochs = 3
    cfg.warmup_portion = 0.1
    cfg.layerwise_lr_decay = 0.8  # The layer_wise decay
    cfg.wd = 0.01  # Weight Decay
    cfg.max_grad_norm = 1.0  # Maximum Gradient Norm
    # The validation frequency = validation frequency * num_updates_in_an_epoch
    cfg.valid_frequency = 0.1
    # Logging frequency = log frequency * num_updates_in_an_epoch
    cfg.log_frequency = 0.1
    return cfg


def base_model_config():
    cfg = CfgNode()
    cfg.backbone = CfgNode()
    cfg.backbone.name = 'google_electra_base'
    cfg.network = MultiModalWithPretrainedTextNN.get_cfg()
    cfg.insert_sep = True              # Whether to insert sep tokens between columns
    cfg.train_stochastic_chunk = True  # Whether to sample a stochastic chunk from the training text
    cfg.test_stochastic_chunk = False  # Whether to use stochastic chunk in testing
    cfg.average_nbest = True           # Whether to average the top performed models. This will usually give us better performance.
    cfg.inference_num_repeat = 1       # Whether to turn on randomness and repeat the inference for multiple times.
    return cfg


def base_learning_config():
    cfg = CfgNode()
    cfg.early_stopping_patience = 10  # Stop if we cannot find a better checkpoint
    cfg.valid_ratio = 0.15  # The ratio of dataset to split for validation
    cfg.stop_metric = 'auto'  # Automatically define the stopping metric
    cfg.log_metrics = 'auto'  # Automatically determine the metrics used in logging
    return cfg


def base_misc_config():
    cfg = CfgNode()
    cfg.seed = 123
    cfg.exp_dir = './autonlp'
    return cfg


def base_cfg():
    cfg = CfgNode()
    cfg.version = 1
    cfg.optimization = base_optimization_config()
    cfg.learning = base_learning_config()
    cfg.preprocessing = base_preprocess_cfg()
    cfg.model = base_model_config()
    cfg.misc = base_misc_config()
    cfg.freeze()
    return cfg


@use_np
def _classification_regression_predict(net, dataloader, problem_type,
                                       has_label=True, extract_embedding=False):
    """

    Parameters
    ----------
    net
        The network
    dataloader
        The dataloader
    problem_type
        Types of the labels
    has_label
        Whether label is used
    extract_embedding
        Whether to extract the embedding

    Returns
    -------
    predictions
        The predictions
    """
    predictions = []
    ctx_l = net.collect_params().list_ctx()
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
            if extract_embedding:
                _, embeddings = net(batch_feature)
                iter_pred_l.append(embeddings)
            else:
                pred = net(batch_feature)
                if problem_type == MULTICLASS or problem_type == BINARY:
                    pred = mx.npx.softmax(pred, axis=-1)
                iter_pred_l.append(pred)
        for pred in iter_pred_l:
            predictions.append(pred.asnumpy())
    predictions = np.concatenate(predictions, axis=0)
    return predictions


def calculate_metric(scorer, ground_truth, predictions, problem_type):
    if problem_type == BINARY and scorer.name == 'roc_auc':
        # For ROC_AUC, we need to feed in the probability of positive class to the scorer.
        return scorer._sign * scorer(ground_truth, predictions[:, 1])
    else:
        return scorer._sign * scorer(ground_truth, predictions)


@use_np
def train_function(args, reporter, train_df_path, tuning_df_path,
                   time_limit, time_start, base_config,
                   problem_type, column_types,
                   feature_columns, label_column,
                   log_metrics, eval_metric,
                   console_log, ignore_warning=False,
                   verbosity=2):
    """

    Parameters
    ----------
    args
        The arguments
    reporter
        Reporter of the HPO scheduler.
        If it is set to None, we won't use the reporter and will just run a single trial.
    train_df_path
        Path of the training dataframe
    tuning_df_path
        Path of the tuning dataframe
    time_limit
        The time limit of calling this function
    time_start
        The starting timestamp of the experiment
    base_config
        Basic configuration
    problem_type
        Type of the problem.
    column_types
        Type of columns
    feature_columns
        The feature columns
    label_column
        Label column
    log_metrics
        Metrics for logging
    eval_metric
        The stopping metric
    console_log
        Whether to log it to console
    ignore_warning
        Whether to ignore warning

    """
    if time_limit is not None:
        start_train_tick = time.time()
        time_left = time_limit - (start_train_tick - time_start)
        if time_left <= 0:
            if reporter is not None:
                reporter.terminate()
            return
    if isinstance(reporter, FakeReporter):
        search_space = args.rand
        task_id = 0
    else:
        search_space = args['search_space']
        task_id = args.task_id
    # Get the log metric scorers
    if isinstance(log_metrics, str):
        log_metrics = [log_metrics]
    # Load the training and tuning data from the parquet file
    train_data = pd.read_parquet(train_df_path)
    tuning_data = pd.read_parquet(tuning_df_path)
    log_metric_scorers = [get_metric(ele) for ele in log_metrics]
    eval_metric_scorer = get_metric(eval_metric)
    greater_is_better = eval_metric_scorer.greater_is_better

    if ignore_warning:
        import warnings
        warnings.filterwarnings("ignore")
    cfg = base_config.clone()
    specified_values = []
    for key in search_space.keys():
        specified_values.append(key)
        specified_values.append(search_space[key])
    cfg.merge_from_list(specified_values)
    exp_dir = cfg.misc.exp_dir
    exp_dir = os.path.join(exp_dir, 'task{}'.format(task_id))
    os.makedirs(exp_dir, exist_ok=True)
    cfg.defrost()
    cfg.misc.exp_dir = exp_dir
    cfg.freeze()
    logger = logging.getLogger()
    set_logger_verbosity(verbosity, logger)
    logging_config(folder=exp_dir, name='training', logger=logger, console=console_log,
                   level=logging.DEBUG,
                   console_level=verbosity2loglevel(verbosity))
    logger.log(15, cfg)

    # Load backbone model
    backbone_model_cls, backbone_cfg, tokenizer, backbone_params_path, _ \
        = get_backbone(cfg.model.backbone.name)
    with open(os.path.join(exp_dir, 'cfg.yml'), 'w') as f:
        f.write(str(cfg))
    text_backbone = backbone_model_cls.from_cfg(backbone_cfg)
    # Build Preprocessor + Preprocess the training dataset + Inference problem type
    # TODO Dynamically cache the preprocessor that has been fitted.
    label_generator = LabelEncoder()
    label_generator.fit(pd.concat([train_data[label_column], tuning_data[label_column]]))
    preprocessor = MultiModalTextFeatureProcessor(column_types=column_types,
                                                  label_column=label_column,
                                                  tokenizer_name=cfg.model.backbone.name,
                                                  label_generator=label_generator,
                                                  cfg=cfg.preprocessing)
    logger.log(25, 'Fitting and transforming the train data...')
    train_dataset = preprocessor.fit_transform(train_data[feature_columns],
                                               train_data[label_column])
    with open(os.path.join(exp_dir, 'preprocessor.pkl'), 'wb') as of:
        pickle.dump(preprocessor, of)
    logger.log(25, f'Done! Preprocessor saved to {os.path.join(exp_dir, "preprocessor.pkl")}')
    logger.log(15, 'Train Data')
    logger.log(15, get_stats_string(preprocessor, train_dataset, is_train=True))
    logger.log(25, 'Process dev set...')
    tuning_dataset = preprocessor.transform(tuning_data[feature_columns],
                                            tuning_data[label_column])
    logger.log(25, 'Done!')
    # Auto Max Length
    if cfg.preprocessing.text.auto_max_length:
        max_length = auto_shrink_max_length(
            train_dataset,
            insert_sep=cfg.model.insert_sep,
            num_text_features=len(preprocessor.text_feature_names),
            auto_max_length_quantile=cfg.preprocessing.text.auto_max_length_quantile,
            round_to=cfg.preprocessing.text.auto_max_length_round_to,
            max_length=cfg.preprocessing.text.max_length)
    else:
        max_length = cfg.preprocessing.text.max_length
    train_stochastic_chunk = cfg.model.train_stochastic_chunk
    test_stochastic_chunk = cfg.model.test_stochastic_chunk
    inference_num_repeat = cfg.model.inference_num_repeat
    logger.log(25, f'Max length for chunking text: {max_length}, '
                   f'Stochastic chunk: Train-{train_stochastic_chunk}/Test-{test_stochastic_chunk}, '
                   f'Test #repeat: {inference_num_repeat}.')
    cls_id, sep_id = get_cls_sep_id(tokenizer)
    train_batchify_fn = MultiModalTextBatchify(
        num_text_inputs=len(preprocessor.text_feature_names),
        num_categorical_inputs=len(preprocessor.categorical_feature_names),
        num_numerical_inputs=len(preprocessor.numerical_feature_names) > 0,
        cls_token_id=cls_id, sep_token_id=sep_id, max_length=max_length,
        mode='train', stochastic_chunk=train_stochastic_chunk,
        insert_sep=cfg.model.insert_sep)
    test_batchify_fn = MultiModalTextBatchify(
        num_text_inputs=len(preprocessor.text_feature_names),
        num_categorical_inputs=len(preprocessor.categorical_feature_names),
        num_numerical_inputs=len(preprocessor.numerical_feature_names) > 0,
        cls_token_id=cls_id, sep_token_id=sep_id, max_length=max_length,
        mode='test', stochastic_chunk=test_stochastic_chunk,
        insert_sep=cfg.model.insert_sep)

    # Get the ground-truth dev labels
    gt_dev_labels = np.array([ele[-1] for ele in tuning_dataset])
    ctx_l = get_mxnet_available_ctx()
    base_batch_size = cfg.optimization.per_device_batch_size
    num_accumulated = int(np.ceil(cfg.optimization.batch_size / (base_batch_size * len(ctx_l))))
    inference_base_batch_size = base_batch_size * cfg.optimization.val_batch_size_mult
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=base_batch_size,
                                  shuffle=True,
                                  batchify_fn=train_batchify_fn)
    dev_dataloader = DataLoader(tuning_dataset,
                                batch_size=inference_base_batch_size,
                                shuffle=False,
                                batchify_fn=test_batchify_fn)
    if problem_type == REGRESSION:
        out_shape = 1
    elif problem_type == MULTICLASS:
        out_shape = len(label_generator.classes_)
    elif problem_type == BINARY:
        assert len(label_generator.classes_) == 2
        out_shape = 2
    else:
        raise NotImplementedError
    net = MultiModalWithPretrainedTextNN(
        text_backbone=text_backbone,
        num_text_features=1,
        num_categorical_features=len(preprocessor.categorical_feature_names),
        num_numerical_features=len(preprocessor.numerical_feature_names) > 0,
        numerical_input_units=None if len(preprocessor.numerical_feature_names) == 0 else len(preprocessor.numerical_feature_names),
        num_categories=preprocessor.categorical_num_categories,
        get_embedding=False,
        cfg=cfg.model.network,
        out_shape=out_shape)
    net.initialize_with_pretrained_backbone(backbone_params_path, ctx=ctx_l)
    net.hybridize()
    num_total_params, num_total_fixed_params = count_parameters(net.collect_params())
    logger.log(25, '#Total Params/Fixed Params={}/{}'.format(num_total_params,
                                                             num_total_fixed_params))
    # Initialize the optimizer
    updates_per_epoch = int(len(train_dataloader) / (num_accumulated * len(ctx_l)))
    optimizer, optimizer_params, max_update \
        = get_optimizer(cfg.optimization,
                        updates_per_epoch=updates_per_epoch)
    valid_interval = math.ceil(cfg.optimization.valid_frequency * updates_per_epoch)
    train_log_interval = math.ceil(cfg.optimization.log_frequency * updates_per_epoch)
    trainer = mx.gluon.Trainer(net.collect_params(),
                               optimizer, optimizer_params,
                               update_on_kvstore=False)
    if 0 < cfg.optimization.layerwise_lr_decay < 1:
        apply_layerwise_decay(net.text_backbone,
                              cfg.optimization.layerwise_lr_decay,
                              backbone_name=cfg.model.backbone.name)
    # Do not apply weight decay to all the LayerNorm and bias
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    params = [p for p in net.collect_params().values() if p.grad_req != 'null']

    # Set grad_req if gradient accumulation is required
    if num_accumulated > 1:
        logger.info('Using gradient accumulation.'
                    ' Global batch size = {}'.format(cfg.optimization.batch_size))
        for p in params:
            p.grad_req = 'add'
        net.collect_params().zero_grad()
    train_loop_dataloader = grouper(repeat(train_dataloader), len(ctx_l))
    log_loss_l = [mx.np.array(0.0, dtype=np.float32, ctx=ctx) for ctx in ctx_l]
    log_num_samples_l = [0 for _ in ctx_l]
    logging_start_tick = time.time()
    nbest = cfg.optimization.nbest
    best_performance_score = []   # Stores the best performing checkpoints
    best_performance_update_idx = []   # Stores the update index that reached the best validation performance
    best_score = None
    mx.npx.waitall()
    no_better_rounds = 0
    report_idx = 0
    start_tick = time.time()
    if time_limit is not None:
        time_limit -= start_tick - time_start
        if time_limit <= 0:
            reporter.terminate()
            return
    best_report_items = None
    for update_idx in range(max_update):
        for accum_idx in range(num_accumulated):
            sample_l = next(train_loop_dataloader)
            loss_l = []
            for i, (sample, ctx) in enumerate(zip(sample_l, ctx_l)):
                feature_batch, label_batch = sample
                feature_batch = move_to_ctx(feature_batch, ctx)
                label_batch = move_to_ctx(label_batch, ctx)
                with mx.autograd.record():
                    pred = net(feature_batch)
                    if problem_type == MULTICLASS or problem_type == BINARY:
                        logits = mx.npx.log_softmax(pred, axis=-1)
                        loss = - mx.npx.pick(logits,
                                             mx.np.expand_dims(label_batch, axis=-1))
                    elif problem_type == REGRESSION:
                        loss = mx.np.square(pred - mx.np.expand_dims(label_batch, axis=-1))
                    loss_l.append(loss.mean() / len(ctx_l) / num_accumulated)
                log_loss_l[i] += loss_l[i] * len(ctx_l) * loss.shape[0] * num_accumulated
                log_num_samples_l[i] += loss.shape[0]
            for loss in loss_l:
                loss.backward()
        # Begin to update
        trainer.allreduce_grads()
        total_norm, ratio, is_finite = clip_grad_global_norm(params, cfg.optimization.max_grad_norm)
        trainer.update(1.0)

        # Clear after update
        if num_accumulated > 1:
            net.collect_params().zero_grad()
        if (update_idx + 1) % train_log_interval == 0:
            log_loss = sum([ele.as_in_ctx(ctx_l[0]) for ele in log_loss_l]).asnumpy()
            log_num_samples = sum(log_num_samples_l)
            logger.log(25,
                '[Iter {}/{}, Epoch {}] train loss={:0.4e}, gnorm={:0.4e}, lr={:0.4e}, #samples processed={},'
                ' #sample per second={:.2f}. ETA={:.2f}min'
                    .format(update_idx + 1, max_update,
                            int(update_idx / updates_per_epoch),
                            log_loss / log_num_samples, total_norm, trainer.learning_rate,
                            log_num_samples,
                            log_num_samples / (time.time() - logging_start_tick),
                            (time.time() - start_tick) / (update_idx + 1)
                            * (max_update - update_idx - 1) / 60))
            logging_start_tick = time.time()
            log_loss_l = [mx.np.array(0.0, dtype=np.float32, ctx=ctx) for ctx in ctx_l]
            log_num_samples_l = [0 for _ in ctx_l]
        if (update_idx + 1) % valid_interval == 0 or (update_idx + 1) == max_update:
            valid_start_tick = time.time()
            dev_predictions = \
                _classification_regression_predict(net,
                                                   dataloader=dev_dataloader,
                                                   problem_type=problem_type,
                                                   has_label=False)
            log_scores = [calculate_metric(scorer, gt_dev_labels,
                                           dev_predictions,
                                           problem_type)
                          for scorer in log_metric_scorers]
            dev_score = calculate_metric(eval_metric_scorer, gt_dev_labels,
                                         dev_predictions,
                                         problem_type)
            valid_time_spent = time.time() - valid_start_tick
            find_better = False
            find_topn_better = False
            if len(best_performance_score) < nbest:
                best_performance_score.append(dev_score)
                best_performance_update_idx.append(update_idx + 1)
                net.save_parameters(
                    os.path.join(exp_dir,
                                 f'nbest_model{len(best_performance_score) - 1}.params'))
                find_topn_better = True
                if best_score is None or greater_is_better and dev_score >= best_score\
                        or (not greater_is_better and dev_score <= best_score):
                    find_better = True
                    net.save_parameters(os.path.join(exp_dir, f'best_model.params'))
                    best_score = dev_score
            else:
                # First try to update the top-K
                if greater_is_better:
                    if dev_score >= min(best_performance_score):
                        find_topn_better = True
                        replace_idx = np.argmin(best_performance_score)
                        best_performance_score[replace_idx] = dev_score
                        best_performance_update_idx[replace_idx] = update_idx + 1
                        net.save_parameters(
                            os.path.join(exp_dir, f'nbest_model{replace_idx}.params'))
                        if dev_score >= best_score:
                            find_better = True
                            net.save_parameters(os.path.join(exp_dir, f'best_model.params'))
                            best_score = dev_score

                else:
                    if dev_score <= max(best_performance_score):
                        find_topn_better = True
                        replace_idx = np.argmax(best_performance_score)
                        best_performance_score[replace_idx] = dev_score
                        best_performance_update_idx[replace_idx] = update_idx + 1
                        net.save_parameters(
                            os.path.join(exp_dir, f'nbest_model{replace_idx}.params'))
                        if dev_score <= best_score:
                            find_better = True
                            net.save_parameters(os.path.join(exp_dir, f'best_model.params'))
                            best_score = dev_score
            if not find_better:
                no_better_rounds += 1
            else:
                no_better_rounds = 0
            mx.npx.waitall()
            loss_string = ', '.join(['{}={:0.4e}'.format(metric.name, score)
                                     for score, metric in zip(log_scores, log_metric_scorers)])
            logger.info('[Iter {}/{}, Epoch {}] valid {}, time spent={:.3f}s,'
                        ' total time spent={:.2f}min. Find new best={}, Find new top-{}={}'.format(
                update_idx + 1, max_update, int(update_idx / updates_per_epoch),
                loss_string, valid_time_spent, (time.time() - start_tick) / 60,
                find_better, nbest, find_topn_better))
            if reporter is not None:
                report_items = [('iteration', update_idx + 1),
                                ('report_idx', report_idx + 1),
                                ('epoch', int(update_idx / updates_per_epoch))] + \
                               [(metric.name, score)
                                for score, metric in zip(log_scores, log_metric_scorers)] + \
                               [('find_better', find_better),
                                ('find_new_topn', find_topn_better),
                                ('time_spent', int(time.time() - start_tick))]
                if eval_metric_scorer._sign < 0:
                    report_items.append(('reward_attr', -dev_score))
                else:
                    report_items.append(('reward_attr', dev_score))
                report_items.append(('eval_metric', eval_metric_scorer.name))
                report_items.append(('exp_dir', exp_dir))
                if find_better:
                    best_report_items = report_items
                reporter(**dict(report_items))
                report_idx += 1
            if no_better_rounds >= cfg.learning.early_stopping_patience:
                logger.info('Early stopping patience reached!')
                break
            total_time_spent = time.time() - start_tick
            if time_limit is not None and total_time_spent > time_limit:
                break
    # Average checkpoints
    best_report_items_dict = dict(best_report_items)
    best_report_items_dict['report_idx'] = report_idx + 1
    reporter(**best_report_items_dict)


def get_recommended_resource(nthreads_per_trial=None,
                             ngpus_per_trial=None) -> Tuple[int, int]:
    """Get the recommended resources.

    Internally, we will try to use GPU whenever it's possible. That means, we will use
    a single GPU for finetuning.

    Parameters
    ----------
    nthreads_per_trial
        The number of threads per trial provided by the user.
    ngpus_per_trial
        The number of GPUs per trial provided by the user.

    Returns
    -------
    nthreads_per_trial
        The recommended resource.
    ngpus_per_trial
    """
    if nthreads_per_trial is None and ngpus_per_trial is None:
        nthreads_per_trial = get_cpu_count()
        ngpus_per_trial = get_gpu_count()
    elif nthreads_per_trial is not None and ngpus_per_trial is None:
        ngpus_per_trial = get_gpu_count()
    elif nthreads_per_trial is None and ngpus_per_trial is not None:
        if ngpus_per_trial != 0:
            num_parallel_jobs = get_gpu_count() // ngpus_per_trial
            nthreads_per_trial = max(get_cpu_count() // num_parallel_jobs, 1)
        else:
            nthreads_per_trial = get_cpu_count()
    nthreads_per_trial = min(nthreads_per_trial, get_cpu_count())
    ngpus_per_trial = min(ngpus_per_trial, get_gpu_count())
    assert nthreads_per_trial > 0 and ngpus_per_trial >= 0,\
        'Invalid number of threads and number of GPUs.'
    return nthreads_per_trial, ngpus_per_trial


class MultiModalTextModel:
    """Learner of the multimodal text data.

    It will be called if the user call `fit()` in TextPrediction tasks.

    It is used for making predictions on new data and viewing information about
    models trained during `fit()`.
    """

    def __init__(self, column_types,
                 feature_columns,
                 label_columns,
                 problem_type,
                 eval_metric,
                 log_metrics,
                 output_directory=None,
                 logger=None):
        """Creates model object.

        Parameters
        ----------
        column_types
            The column types.
        feature_columns
            Name of the feature columns
        label_columns
            Name of the label columns.
        problem_type
            Type of the problem
        eval_metric
            The evaluation metric
        log_metrics
            The metrics for logging
        output_directory
            The output directory to save the model
        logger
            The logger
        """
        super(MultiModalTextModel, self).__init__()
        self._base_config = base_cfg()
        self._base_config.defrost()
        if output_directory is not None:
            self._base_config.misc.exp_dir = output_directory
        self._base_config.misc.exp_dir = os.path.abspath(self._base_config.misc.exp_dir)
        self._base_config.freeze()
        self._output_directory = self._base_config.misc.exp_dir
        self._column_types = column_types
        self._eval_metric = eval_metric
        self._log_metrics = log_metrics
        self._logger = logger

        self._label_columns = label_columns
        self._feature_columns = feature_columns
        self._problem_type = problem_type

        # Need to be set in the train call
        self._net = None  # Network for training and inference
        self._embed_net = None  # Network for extract the embedding
        self._feature_generator = None  # The feature generator
        self._multimodal_preprocessor = None  # The inner preprocessor
        self._config = None
        self._results = None

    @property
    def output_directory(self):
        """ Get the output directory. The trained model and the training logs
        will be saved to this folder """
        return self._output_directory

    @property
    def label_columns(self):
        """Name of the label columns"""
        return self._label_columns

    @property
    def problem_type(self):
        """Types of the problem"""
        return self._problem_type

    @property
    def feature_columns(self):
        """Name of the features"""
        return self._feature_columns

    @property
    def base_config(self):
        """The basic configuration. Internally, we will fill values in the base config by values
        in the search space."""
        return self._base_config

    @property
    def results(self):
        """Results of the final model"""
        return self._results

    @property
    def config(self):
        """The configuration of the final trained model."""
        return self._config

    @property
    def net(self):
        return self._net

    def train(self, train_data, tuning_data,
              num_cpus=None,
              num_gpus=None,
              time_limit=None,
              hpo_params=None,
              search_space=None,
              plot_results=False,
              console_log=True,
              ignore_warning=False,
              verbosity=2):
        """The train function.

        Parameters
        ----------
        train_data
            The training data
        tuning_data
            The tuning data
        num_cpus
            Number of CPUs for each trial
        num_gpus
            Number of GPUs for each trial
        time_limit
            The time limits
        hpo_params
            Parameters of the HPO algorithms. For example, the scheduling
            algorithm, scheduling backend, HPO algorithm.
        search_space
            The search space options
        plot_results
            Whether to plot results or not
        console_log
            Whether to log into the console
        ignore_warning
            Whether to ignore the warning
        verbosity
            Verbosity
        """
        start_tick = time.time()
        assert len(self._label_columns) == 1, 'Currently, we only support single label.'
        # TODO(sxjscience) Try to support S3
        os.makedirs(self._output_directory, exist_ok=True)
        if search_space is None:
            search_space = \
                ag_text_presets.create('default')['models']['MultimodalTextModel']['search_space']
        search_space_reg = args(search_space=space.Dict(**search_space))
        # Scheduler and searcher for HPO
        if hpo_params is None:
            hpo_params = ag_text_presets.create('default')['hpo_params']
        scheduler_options = hpo_params['scheduler_options']
        num_cpus, num_gpus = get_recommended_resource(num_cpus, num_gpus)
        self._logger.log(25, f"The GluonNLP V0 backend is used. "
                             f"We will use {num_cpus} cpus and "
                             f"{num_gpus} gpus to train each trial.")
        if scheduler_options is None:
            scheduler_options = dict()
        scheduler_options = compile_scheduler_options_v2(
            scheduler_options=scheduler_options,
            search_strategy=hpo_params['search_strategy'],
            search_options=hpo_params['search_options'],
            nthreads_per_trial=num_cpus,
            ngpus_per_trial=num_gpus,
            checkpoint=os.path.join(self._output_directory, 'checkpoint.ag'),
            num_trials=hpo_params['num_trials'],
            time_out=time_limit,
            resume=False,
            visualizer=scheduler_options.get('visualizer'),
            time_attr='report_idx',
            reward_attr='reward_attr',
            dist_ip_addrs=scheduler_options.get('dist_ip_addrs'))
        # Create a temporary cache file. The internal train function will load the
        # temporary cache.
        os.makedirs(os.path.join(self._output_directory, 'data_cache'), exist_ok=True)
        train_df_path = os.path.join(self._output_directory, 'data_cache',
                                     'cache_train_dataframe.pq')
        tuning_df_path = os.path.join(self._output_directory,  'data_cache',
                                      'cache_tuning_dataframe.pq')
        train_data.to_parquet(train_df_path)
        tuning_data.to_parquet(tuning_df_path)
        train_fn = search_space_reg(functools.partial(train_function,
                                                      train_df_path=train_df_path,
                                                      time_limit=time_limit,
                                                      time_start=start_tick,
                                                      tuning_df_path=tuning_df_path,
                                                      base_config=self.base_config,
                                                      problem_type=self.problem_type,
                                                      column_types=self._column_types,
                                                      feature_columns=self._feature_columns,
                                                      label_column=self._label_columns[0],
                                                      log_metrics=self._log_metrics,
                                                      eval_metric=self._eval_metric,
                                                      console_log=console_log,
                                                      ignore_warning=ignore_warning,
                                                      verbosity=verbosity))
        if scheduler_options['num_trials'] == 1:
            train_fn(train_fn.args['search_space'],
                     train_fn.args['_default_config'])
            best_model_saved_dir_path = os.path.join(self._output_directory,'task0'.format(best_task_id))
            cfg_path = os.path.join(self._output_directory, 'task0', 'cfg.yml')
            cfg = self.base_config.clone_merge(cfg_path)
        else:
            force_forkserver()
            scheduler_cls = schedulers[scheduler_options['searcher']]
            # Create scheduler, run HPO experiment
            scheduler = scheduler_cls(train_fn, **scheduler_options)
            scheduler.run()
            scheduler.join_jobs()
            if len(scheduler.config_history) == 0:
                raise RuntimeError('No training job has been completed! '
                                   'There are two possibilities: '
                                   '1) The time_limits is too small, '
                                   'or 2) There are some internal errors in AutoGluon. '
                                   'For the first case, you can increase the time_limits or set it to '
                                   'None, e.g., setting "predictor.fit(..., time_limit=None). To '
                                   'further investigate the root cause, you can also try to set the '
                                   '"verbosity=3" and try again, i.e., predictor.set_verbosity(3).')
            best_config = scheduler.get_best_config()
            self._logger.log(25, 'Results=', scheduler.searcher._results)
            self._logger.log(25, 'Best_config={}'.format(best_config))
            best_task_id = scheduler.get_best_task_id()
            best_model_saved_dir_path = os.path.join(self._output_directory,
                                                     'task{}'.format(best_task_id))
            best_cfg_path = os.path.join(best_model_saved_dir_path, 'cfg.yml')
            cfg = self.base_config.clone_merge(best_cfg_path)
            if plot_results is None:
                if in_ipynb():
                    plot_results = True
                else:
                    plot_results = False
            if plot_results:
                plot_training_curves = os.path.join(self._output_directory,
                                                    'plot_training_curves.png')
                scheduler.get_training_curves(filename=plot_training_curves,
                                              plot=plot_results,
                                              use_legend=True)
            self._results = dict()
            self._results.update(best_reward=scheduler.get_best_reward(),
                                 best_config=scheduler.get_best_config(),
                                 total_time=time.time() - start_tick,
                                 metadata=scheduler.metadata,
                                 training_history=scheduler.training_history,
                                 config_history=scheduler.config_history,
                                 reward_attr=scheduler._reward_attr,
                                 config=cfg)
        # Consider to move this to a separate predictor
        self._config = cfg
        backbone_model_cls, backbone_cfg, tokenizer, backbone_params_path, _ \
            = get_backbone(cfg.model.backbone.name)
        text_backbone = backbone_model_cls.from_cfg(backbone_cfg)
        preprocessor = TabularBasicBERTPreprocessor(tokenizer=tokenizer,
                                                    column_properties=self._column_properties,
                                                    label_columns=self._label_columns,
                                                    max_length=cfg.model.preprocess.max_length,
                                                    merge_text=cfg.model.preprocess.merge_text)
        self._preprocessor = preprocessor
        net = BERTForTabularBasicV1(text_backbone=text_backbone,
                                    feature_field_info=preprocessor.feature_field_info(),
                                    label_shape=self._label_shapes[0],
                                    cfg=cfg.model.network)
        net.hybridize()
        ctx_l = get_mxnet_available_ctx()
        net.load_parameters(os.path.join(best_model_saved_dir_path, 'best_model.params'),
                            ctx=ctx_l)
        self._net = net
        mx.npx.waitall()
        self.save(self._output_directory)

    def evaluate(self, valid_data, metrics=None, return_type='list'):
        """ Report the predictive performance evaluated for a given dataset.

        Parameters
        ----------
        valid_data : str or :class:`TabularDataset` or `pandas.DataFrame`
            This Dataset must also contain the label-column with the same column-name as specified during `fit()`.
            If str is passed, `valid_data` will be loaded using the str value as the file path.
        metrics : str or List[str] or None
            Name of metric or a list of names of metrics to report.
            If it is not given, we will return the score of the stored eval_metric.
        return_type :
            Can be list or dict

        Returns
        -------
        List of metric scores
        """
        if isinstance(metrics, str):
            metrics = [metrics]
        assert self.net is not None
        if not isinstance(valid_data, pd.DataFrame):
            valid_data = load_pd.load(valid_data)
        valid_data = valid_data[self._feature_columns + self._label_columns]
        ground_truth = np.array(valid_data.table[self._label_columns[0]].apply(
            self._column_properties[self._label_columns[0]].transform))
        if self._problem_types[0] == _C.CLASSIFICATION:
            predictions = self.predict_proba(valid_data)
        else:
            predictions = self.predict(valid_data)
        metric_scores = {metric: calculate_metric(get_metric(metric), ground_truth, predictions,
                                                  self.problem_types[0]) for metric in metrics}
        return metric_scores

    def _internal_predict(self, test_data, get_original_labels=True, get_probabilities=False):
        assert self.net is not None
        assert self.config is not None
        if not isinstance(test_data, TabularDataset):
            if isinstance(test_data, (list, dict)):
                test_data = pd.DataFrame(test_data)
            test_data = TabularDataset(test_data,
                                       columns=self._feature_columns,
                                       column_properties=self._column_properties)
        processed_test = self._preprocessor.process_test(test_data)
        inference_batch_size = self.config.optimization.per_device_batch_size \
                               * self.config.optimization.val_batch_size_mult
        test_dataloader = DataLoader(processed_test,
                                     batch_size=inference_batch_size,
                                     shuffle=False,
                                     batchify_fn=self._preprocessor.batchify(is_test=True))
        test_predictions = _classification_regression_predict(self._net,
                                                              dataloader=test_dataloader,
                                                              problem_type=self._problem_types[0],
                                                              has_label=False)
        if self._problem_types[0] == _C.CLASSIFICATION:
            if get_probabilities:
                return test_predictions
            else:
                test_predictions = test_predictions.argmax(axis=-1)
                if get_original_labels:
                    test_predictions = np.array(
                        list(map(self._column_properties[self._label_columns[0]].inv_transform,
                                 test_predictions)))
        return test_predictions

    @property
    def class_labels(self):
        """The original name of the class labels.

        For example, the tabular data may contain classes equal to
        "entailment", "contradiction", "neutral". Internally, these will be converted to
        0, 1, 2, ...

        This function returns the original names of these raw labels.

        Returns
        -------
        ret
            List that contain the class names
        """
        if self.problem_type != MULTICLASS or self.problem_type != BINARY:
            warnings.warn('Accessing class names for a non-classification problem. Return None.')
            return None
        else:
            raise NotImplementedError

    def predict_proba(self, test_data):
        """Predict class probabilities instead of class labels (for classification tasks).

        Parameters
        ----------
        test_data : `pandas.DataFrame`, `autogluon.tabular.TabularDataset`, or str
            The test data to get predictions for. Can be DataFrame/Dataset or a file that can
            be loaded into DataFrame/Dataset.

        Returns
        -------
        probabilities : array
            The predicted class probabilities for each sample.
            Shape of this array is (#Samples, num_class).
            Here, the i-th number means the probability of belonging to the i-th class.
            You can access the class names by calling `self.class_names`.
        """
        assert self.problem_type == MULTICLASS or self.problem_type == BINARY
        return self._internal_predict(test_data,
                                      get_original_labels=False,
                                      get_probabilities=True)

    def predict(self, test_data, get_original_labels=True):
        """Make predictions on new data.

        Parameters
        ----------
        test_data : `pandas.DataFrame`, `autogluon.tabular.TabularDataset`, or str
            The test data to get predictions for. Can be DataFrame/Dataset or a file that can be loaded into DataFrame/Dataset.
        get_original_labels : bool, default = True
            Whether or not predictions should be formatted in terms of the original labels.
            For example, the labels might be "entailment" or "not_entailment" and predictions could either be of this form (if `True`) or integer-indices corresponding to these classes (if `False`).

        Returns
        -------
        predictions : array
            The predictions for each sample. Shape of this array is (#Samples,).
        """
        return self._internal_predict(test_data,
                                      get_original_labels=get_original_labels,
                                      get_probabilities=False)

    def save(self, dir_path):
        """Save this model to disk.

        Parameters
        ----------
        dir_path : str
            Directory where the model should be saved.
        """
        os.makedirs(dir_path, exist_ok=True)
        self.net.save_parameters(os.path.join(dir_path, 'net.params'))
        with open(os.path.join(dir_path, 'cfg.yml'), 'w') as of:
            of.write(self.config.dump())
        # Save an additional assets about the parsed dataset information
        with open(os.path.join(dir_path, 'assets.json'), 'w') as of:
            json.dump(
                {
                    'problem_type': self._problem_type,
                    'label_columns': self._label_columns,
                    'feature_columns': self._feature_columns,
                    'column_types': self._column_types,
                    'version': version.__version__,
                }, of, ensure_ascii=True)

    def cuda(self):
        """Try to use CUDA for inference"""
        self._net.collect_params().reset_ctx(mx.gpu())

    def cpu(self):
        """Switch to use CPU for inference"""
        self._net.collect_params().reset_ctx(mx.cpu())

    @classmethod
    def load(cls, dir_path: str):
        """Load a model object previously produced by `fit()` from disk and return this object.
           It is highly recommended the predictor be loaded with the exact AutoGluon version
           it was fit with.

        Parameters
        ----------
        dir_path
            Path to directory where this model was previously saved.

        Returns
        -------
        model
            A `BertForTextPredictionBasic` object that can be used for making predictions on new data.
        """
        # TODO In general, we will need to support compatible version check
        loaded_config = base_cfg().clone_merge(os.path.join(dir_path, 'cfg.yml'))
        with open(os.path.join(dir_path, 'assets.json'), 'r') as f:
            assets = json.load(f)
        label_columns = assets['label_columns']
        feature_columns = assets['feature_columns']
        label_shapes = assets['label_shapes']
        problem_type = assets['problem_type']
        column_types = assets['column_types']
        backbone_model_cls, backbone_cfg, tokenizer, backbone_params_path, _ \
            = get_backbone(loaded_config.model.backbone.name)
        # Initialize the preprocessor
        text_backbone = backbone_model_cls.from_cfg(backbone_cfg)
        net = BERTForTabularBasicV1(text_backbone=text_backbone,
                                    feature_field_info=preprocessor.feature_field_info(),
                                    label_shape=label_shapes[0],
                                    cfg=loaded_config.model.network)
        net.hybridize()
        ctx_l = get_mxnet_available_ctx()
        net.load_parameters(os.path.join(dir_path, 'net.params'), ctx=ctx_l)
        model = cls(label_columns=label_columns,
                    feature_columns=feature_columns,
                    problem_type=problem_type,
                    eval_metric=eval_metric,
                    log_metrics=log_metrics,
                    base_config=loaded_config)
        model._net = net
        model._preprocessor = preprocessor
        model._config = loaded_config
        return model

    def extract_embedding(self, data):
        """Extract the embedding from the pretrained model.

        Returns
        -------
        embeddings
            The output embeddings will have shape
            (#samples, embedding_dim)
        """
        if not isinstance(data, TabularDataset):
            if isinstance(data, (list, dict)):
                data = pd.DataFrame(data)
            data = TabularDataset(data,
                                  columns=self._feature_columns,
                                  column_properties=self._column_properties)
        processed_data = self._preprocessor.process_test(data)
        inference_batch_size = self.config.optimization.per_device_batch_size \
                               * self.config.optimization.val_batch_size_mult
        dataloader = DataLoader(processed_data,
                                batch_size=inference_batch_size,
                                shuffle=False,
                                batchify_fn=self._preprocessor.batchify(is_test=True))
        if self._embed_net is None:
            embed_net = BERTForTabularBasicV1(
                text_backbone=self.net.text_backbone,
                feature_field_info=self._preprocessor.feature_field_info(),
                label_shape=self.label_shapes[0],
                cfg=self.config.model.network,
                get_embedding=True,
                params=self.net.collect_params(),
                prefix='embed_net_')
            embed_net.hybridize()
            self._embed_net = embed_net
        embeddings = _classification_regression_predict(self._embed_net,
                                                        dataloader=dataloader,
                                                        problem_type=self._problem_types[0],
                                                        has_label=False,
                                                        extract_embedding=True)
        return embeddings
