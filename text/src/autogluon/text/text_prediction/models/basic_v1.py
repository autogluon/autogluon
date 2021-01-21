import numpy as np
import os
import math
import logging
import pandas as pd
import warnings
import time
import json
import functools
import tqdm
import mxnet as mx
from mxnet.util import use_np
from mxnet.lr_scheduler import PolyScheduler, CosineScheduler
from mxnet.gluon.data import DataLoader
from autogluon_contrib_nlp.models import get_backbone
from autogluon_contrib_nlp.lr_scheduler import InverseSquareRootScheduler
from autogluon_contrib_nlp.utils.config import CfgNode
from autogluon_contrib_nlp.utils.misc import logging_config, grouper,\
    count_parameters, repeat, get_mxnet_available_ctx
from autogluon_contrib_nlp.utils.parameter import move_to_ctx, clip_grad_global_norm
from autogluon.core import args, space
from autogluon.core.task.base import compile_scheduler_options
from autogluon.core.task.base.base_task import schedulers
from autogluon.core.metrics import get_metric, Scorer
from autogluon.core.utils.multiprocessing_utils import force_forkserver

from .. import constants as _C
from ..column_property import get_column_property_metadata, get_column_properties_from_metadata
from ..preprocessing import TabularBasicBERTPreprocessor
from ..modules.basic_prediction import BERTForTabularBasicV1
from ..dataset import TabularDataset
from ... import version


@use_np
def get_optimizer(cfg, updates_per_epoch):
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
    cfg.valid_frequency = 0.1
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
    cfg.early_stopping_patience = 10  # Stop if we cannot find a better checkpoint
    cfg.valid_ratio = 0.15      # The ratio of dataset to split for validation
    cfg.stop_metric = 'auto'    # Automatically define the stopping metric
    cfg.log_metrics = 'auto'    # Automatically determine the metrics used in logging
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
                if problem_type == _C.CLASSIFICATION:
                    pred = mx.npx.softmax(pred, axis=-1)
                iter_pred_l.append(pred)
        for pred in iter_pred_l:
            predictions.append(pred.asnumpy())
    predictions = np.concatenate(predictions, axis=0)
    return predictions


def calculate_metric(scorer, ground_truth, predictions, problem_type):
    if problem_type == _C.CLASSIFICATION and scorer.name == 'roc_auc':
        # For ROC_AUC, we need to feed in the probability of positive class to the scorer.
        return scorer._sign * scorer(ground_truth, predictions[:, 1])
    else:
        return scorer._sign * scorer(ground_truth, predictions)


@use_np
def train_function(args, reporter, train_df_path, tuning_df_path,
                   time_limits, time_start, base_config, problem_types,
                   column_properties, label_columns, label_shapes,
                   log_metrics, stopping_metric, console_log,
                   ignore_warning=False):
    if time_limits is not None:
        start_train_tick = time.time()
        time_left = time_limits - (start_train_tick - time_start)
        if time_left <= 0:
            reporter.terminate()
            return
    import os
    # Get the log metric scorers
    if isinstance(log_metrics, str):
        log_metrics = [log_metrics]
    # Load the training and tuning data from the parquet file
    train_data = pd.read_parquet(train_df_path)
    tuning_data = pd.read_parquet(tuning_df_path)
    log_metric_scorers = [get_metric(ele) for ele in log_metrics]
    stopping_metric_scorer = get_metric(stopping_metric)
    greater_is_better = stopping_metric_scorer.greater_is_better
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_DYNAMIC'] = 'FALSE'
    if ignore_warning:
        import warnings
        warnings.filterwarnings("ignore")
    search_space = args['search_space']
    cfg = base_config.clone()
    specified_values = []
    for key in search_space:
        specified_values.append(key)
        specified_values.append(search_space[key])
    cfg.merge_from_list(specified_values)
    exp_dir = cfg.misc.exp_dir
    if reporter is not None:
        # When the reporter is not None,
        # we create the saved directory based on the task_id + time
        task_id = args.task_id
        exp_dir = os.path.join(exp_dir, 'task{}'.format(task_id))
        os.makedirs(exp_dir, exist_ok=True)
        cfg.defrost()
        cfg.misc.exp_dir = exp_dir
        cfg.freeze()
    logger = logging.getLogger()
    logging_config(folder=exp_dir, name='training', logger=logger, console=console_log)
    logger.info(cfg)
    # Load backbone model
    backbone_model_cls, backbone_cfg, tokenizer, backbone_params_path, _ \
        = get_backbone(cfg.model.backbone.name)
    with open(os.path.join(exp_dir, 'cfg.yml'), 'w') as f:
        f.write(str(cfg))
    text_backbone = backbone_model_cls.from_cfg(backbone_cfg)
    # Build Preprocessor + Preprocess the training dataset + Inference problem type
    # TODO Move preprocessor + Dataloader to outer loop to better cache the dataloader
    preprocessor = TabularBasicBERTPreprocessor(tokenizer=tokenizer,
                                                column_properties=column_properties,
                                                label_columns=label_columns,
                                                max_length=cfg.model.preprocess.max_length,
                                                merge_text=cfg.model.preprocess.merge_text)
    logger.info('Process training set...')
    processed_train = preprocessor.process_train(train_data)
    logger.info('Done!')
    logger.info('Process dev set...')
    processed_dev = preprocessor.process_test(tuning_data)
    logger.info('Done!')
    label = label_columns[0]
    # Get the ground-truth dev labels
    gt_dev_labels = np.array(tuning_data[label].apply(column_properties[label].transform))
    ctx_l = get_mxnet_available_ctx()
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
                                label_shape=label_shapes[0],
                                cfg=cfg.model.network)
    net.initialize_with_pretrained_backbone(backbone_params_path, ctx=ctx_l)
    net.hybridize()
    num_total_params, num_total_fixed_params = count_parameters(net.collect_params())
    logger.info('#Total Params/Fixed Params={}/{}'.format(num_total_params,
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
    best_performance_score = None
    mx.npx.waitall()
    no_better_rounds = 0
    report_idx = 0
    start_tick = time.time()
    if time_limits is not None:
        time_limits -= start_tick - time_start
        if time_limits <= 0:
            reporter.terminate()
            return
    best_report_items = None
    for update_idx in tqdm.tqdm(range(max_update), disable=None):
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
                    if problem_types[0] == _C.CLASSIFICATION:
                        logits = mx.npx.log_softmax(pred, axis=-1)
                        loss = - mx.npx.pick(logits, label_batch[0])
                    elif problem_types[0] == _C.REGRESSION:
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
            logger.info(
                '[Iter {}/{}, Epoch {}] train loss={}, gnorm={}, lr={}, #samples processed={},'
                ' #sample per second={}'
                    .format(update_idx + 1, max_update,
                            int(update_idx / updates_per_epoch),
                            log_loss / log_num_samples, total_norm, trainer.learning_rate,
                            log_num_samples,
                            log_num_samples / (time.time() - logging_start_tick)))
            logging_start_tick = time.time()
            log_loss_l = [mx.np.array(0.0, dtype=np.float32, ctx=ctx) for ctx in ctx_l]
            log_num_samples_l = [0 for _ in ctx_l]
        if (update_idx + 1) % valid_interval == 0 or (update_idx + 1) == max_update:
            valid_start_tick = time.time()
            dev_predictions = \
                _classification_regression_predict(net, dataloader=dev_dataloader,
                                                   problem_type=problem_types[0],
                                                   has_label=False)
            log_scores = [calculate_metric(scorer, gt_dev_labels, dev_predictions, problem_types[0])
                          for scorer in log_metric_scorers]
            dev_score = calculate_metric(stopping_metric_scorer, gt_dev_labels, dev_predictions,
                                         problem_types[0])
            valid_time_spent = time.time() - valid_start_tick

            if best_performance_score is None or \
                    (greater_is_better and dev_score >= best_performance_score) or \
                    (not greater_is_better and dev_score <= best_performance_score):
                find_better = True
                no_better_rounds = 0
                best_performance_score = dev_score
                net.save_parameters(os.path.join(exp_dir, 'best_model.params'))
            else:
                find_better = False
                no_better_rounds += 1
            mx.npx.waitall()
            loss_string = ', '.join(['{}={}'.format(metric.name, score)
                                     for score, metric in zip(log_scores, log_metric_scorers)])
            logger.info('[Iter {}/{}, Epoch {}] valid {}, time spent={},'
                         ' total_time={:.2f}min'.format(
                update_idx + 1, max_update, int(update_idx / updates_per_epoch),
                loss_string, valid_time_spent, (time.time() - start_tick) / 60))
            report_items = [('iteration', update_idx + 1),
                            ('report_idx', report_idx + 1),
                            ('epoch', int(update_idx / updates_per_epoch))] +\
                           [(metric.name, score)
                            for score, metric in zip(log_scores, log_metric_scorers)] + \
                           [('find_better', find_better),
                            ('time_spent', int(time.time() - start_tick))]
            total_time_spent = time.time() - start_tick

            if stopping_metric_scorer._sign < 0:
                report_items.append(('reward_attr', -dev_score))
            else:
                report_items.append(('reward_attr', dev_score))
            report_items.append(('eval_metric', stopping_metric_scorer.name))
            report_items.append(('exp_dir', exp_dir))
            if find_better:
                best_report_items = report_items
            reporter(**dict(report_items))
            report_idx += 1
            if no_better_rounds >= cfg.learning.early_stopping_patience:
                logger.info('Early stopping patience reached!')
                break
            if time_limits is not None and total_time_spent > time_limits:
                break

    best_report_items_dict = dict(best_report_items)
    best_report_items_dict['report_idx'] = report_idx + 1
    reporter(**best_report_items_dict)

@use_np
class BertForTextPredictionBasic:
    """A model object returned by `fit()` in TextPrediction tasks. 
       Use for making predictions on new data and viewing information about models trained during `fit()`.
    """
    
    def __init__(self, column_properties, label_columns, feature_columns,
                 label_shapes, problem_types, stopping_metric, log_metrics,
                 output_directory=None, logger=None, base_config=None, search_space=None):
        """Creates model object.

        Parameters
        ----------
        column_properties
            The column properties.
        label_columns
            Label columns.
        feature_columns
        label_shapes
        problem_types
        stopping_metric
        log_metrics
        output_directory
        logger
        base_config
            The basic configuration that the search space will be based upon.
        search_space
            The hyperparameter search space.
        """
        super(BertForTextPredictionBasic, self).__init__()
        if base_config is None:
            self._base_config = base_cfg()
        else:
            self._base_config = base_cfg().clone_merge(base_config)
        self._base_config.defrost()
        if output_directory is not None:
            self._base_config.misc.exp_dir = output_directory
        else:
            output_directory = self._base_config.misc.exp_dir
        self._base_config.misc.exp_dir = os.path.abspath(self._base_config.misc.exp_dir)
        self._base_config.freeze()
        if search_space is None:
            self._search_space = dict()
        else:
            assert isinstance(search_space, dict)
            self._search_space = search_space
        self._column_properties = column_properties
        self._stopping_metric = stopping_metric
        self._log_metrics = log_metrics
        self._logger = logger
        self._output_directory = output_directory

        self._label_columns = label_columns
        self._feature_columns = feature_columns
        self._label_shapes = label_shapes
        self._problem_types = problem_types

        # Need to be set in the fit call
        self._net = None
        self._embed_net = None
        self._preprocessor = None
        self._config = None
        self._results = None

    @property
    def label_columns(self):
        return self._label_columns

    @property
    def label_shapes(self):
        return self._label_shapes

    @property
    def problem_types(self):
        return self._problem_types

    @property
    def feature_columns(self):
        return self._feature_columns

    @property
    def search_space(self):
        return self._search_space

    @property
    def base_config(self):
        return self._base_config

    @property
    def results(self):
        return self._results

    @property
    def config(self):
        return self._config

    @property
    def net(self):
        return self._net

    @staticmethod
    def default_config():
        """Get the default configuration

        Returns
        -------
        cfg
            The configuration specified by the key
        """
        return base_cfg()

    def train(self, train_data, tuning_data, resource,
              time_limits=None,
              search_strategy='random',
              search_options=None,
              scheduler_options=None,
              num_trials=None,
              plot_results=False,
              console_log=True,
              ignore_warning=True):
        force_forkserver()
        start_tick = time.time()
        logging_config(folder=self._output_directory, name='main',
                       console=console_log,
                       logger=self._logger)
        assert len(self._label_columns) == 1
        # TODO(sxjscience) Try to support S3
        os.makedirs(self._output_directory, exist_ok=True)
        search_space_reg = args(search_space=space.Dict(**self.search_space))
        # Scheduler and searcher for HPO
        if scheduler_options is None:
            scheduler_options = dict()
        scheduler_options = compile_scheduler_options(
            scheduler_options=scheduler_options,
            search_strategy=search_strategy,
            search_options=search_options,
            nthreads_per_trial=resource['num_cpus'],
            ngpus_per_trial=resource['num_gpus'],
            checkpoint=os.path.join(self._output_directory, 'checkpoint.ag'),
            num_trials=num_trials,
            time_out=time_limits,
            resume=False,
            visualizer=scheduler_options.get('visualizer'),
            time_attr='report_idx',
            reward_attr='reward_attr',
            dist_ip_addrs=scheduler_options.get('dist_ip_addrs'))
        # Create a temporary cache file and then ask the inner function to load the
        # temporary cache.
        train_df_path = os.path.join(self._output_directory, 'cache_train_dataframe.pq')
        tuning_df_path = os.path.join(self._output_directory, 'cache_tuning_dataframe.pq')
        train_data.table.to_parquet(train_df_path)
        tuning_data.table.to_parquet(tuning_df_path)
        train_fn = search_space_reg(functools.partial(train_function,
                                                      train_df_path=train_df_path,
                                                      time_limits=time_limits,
                                                      time_start=start_tick,
                                                      tuning_df_path=tuning_df_path,
                                                      base_config=self.base_config,
                                                      problem_types=self.problem_types,
                                                      column_properties=self._column_properties,
                                                      label_columns=self._label_columns,
                                                      label_shapes=self._label_shapes,
                                                      log_metrics=self._log_metrics,
                                                      stopping_metric=self._stopping_metric,
                                                      console_log=console_log,
                                                      ignore_warning=ignore_warning))
        scheduler_cls = schedulers[search_strategy.lower()]
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
                               'None, e.g., setting "TextPrediction.fit(..., time_limits=None). To '
                               'further investigate the root cause, you can also try to train with '
                               '"verbosity=3", i.e., TextPrediction.fit(..., verbosity=3).')
        best_config = scheduler.get_best_config()
        self._logger.info('Results=', scheduler.searcher._results)
        self._logger.info('Best_config={}'.format(best_config))
        best_task_id = scheduler.get_best_task_id()
        best_model_saved_dir_path = os.path.join(self._output_directory,
                                                 'task{}'.format(best_task_id))
        best_cfg_path = os.path.join(best_model_saved_dir_path, 'cfg.yml')
        cfg = self.base_config.clone_merge(best_cfg_path)
        self._results = dict()
        self._results.update(best_reward=scheduler.get_best_reward(),
                             best_config=scheduler.get_best_config(),
                             total_time=time.time() - start_tick,
                             metadata=scheduler.metadata,
                             training_history=scheduler.training_history,
                             config_history=scheduler.config_history,
                             reward_attr=scheduler._reward_attr,
                             config=cfg)
        if plot_results:
            plot_training_curves = os.path.join(self._output_directory, 'plot_training_curves.png')
            scheduler.get_training_curves(filename=plot_training_curves, plot=plot_results,
                                          use_legend=True)
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

    def evaluate(self, valid_data, metrics):
        """ Report the predictive performance evaluated for a given dataset.
            
            Parameters
            ----------
            valid_data : str or :class:`TabularDataset` or `pandas.DataFrame`
                This Dataset must also contain the label-column with the same column-name as specified during `fit()`.
                If str is passed, `valid_data` will be loaded using the str value as the file path.
            metrics : List[str]
                A list of names of metrics to report.
 
            Returns
            -------
            Dict mapping metric -> score calculated over the given dataset.
        """
        if isinstance(metrics, str):
            metrics = [metrics]
        assert self.net is not None
        if not isinstance(valid_data, TabularDataset):
            valid_data = TabularDataset(valid_data,
                                        columns=self._feature_columns + self._label_columns,
                                        column_properties=self._column_properties)
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
        inference_batch_size = self.config.optimization.per_device_batch_size\
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
        if self._problem_types[0] != _C.CLASSIFICATION:
            warnings.warn('Accessing class names for a non-classification problem. Return None.')
            return None
        else:
            return self._column_properties[self._label_columns[0]].categories

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
        assert self.problem_types[0] == _C.CLASSIFICATION
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
        with open(os.path.join(dir_path, 'column_metadata.json'), 'w') as of:
            json.dump(get_column_property_metadata(self._column_properties),
                      of, ensure_ascii=True)
        # Save an additional assets about the parsed dataset information
        with open(os.path.join(dir_path, 'assets.json'), 'w') as of:
            json.dump(
                {
                    'label_columns': self._label_columns,
                    'label_shapes': self._label_shapes,
                    'problem_types': self._problem_types,
                    'feature_columns': self._feature_columns,
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
           It is highly recommended the predictor be loaded with the exact AutoGluon version it was fit with.


        Parameters
        ----------
        dir_path
            Path to directory where this model was previously saved.
        use_gpu
            Whether try to use GPU if possible.

        Returns
        -------
        model
            A `BertForTextPredictionBasic` object that can be used for making predictions on new data.
        """
        loaded_config = cls.default_config().clone_merge(os.path.join(dir_path, 'cfg.yml'))
        with open(os.path.join(dir_path, 'assets.json'), 'r') as f:
            assets = json.load(f)
        label_columns = assets['label_columns']
        feature_columns = assets['feature_columns']
        label_shapes = assets['label_shapes']
        problem_types = assets['problem_types']
        column_properties = get_column_properties_from_metadata(
            os.path.join(dir_path, 'column_metadata.json'))
        backbone_model_cls, backbone_cfg, tokenizer, backbone_params_path, _ \
            = get_backbone(loaded_config.model.backbone.name)
        # Initialize the preprocessor
        preprocessor = TabularBasicBERTPreprocessor(
            tokenizer=tokenizer,
            column_properties=column_properties,
            label_columns=label_columns,
            max_length=loaded_config.model.preprocess.max_length,
            merge_text=loaded_config.model.preprocess.merge_text)
        text_backbone = backbone_model_cls.from_cfg(backbone_cfg)
        net = BERTForTabularBasicV1(text_backbone=text_backbone,
                                    feature_field_info=preprocessor.feature_field_info(),
                                    label_shape=label_shapes[0],
                                    cfg=loaded_config.model.network)
        net.hybridize()
        ctx_l = get_mxnet_available_ctx()
        net.load_parameters(os.path.join(dir_path, 'net.params'), ctx=ctx_l)
        model = cls(column_properties=column_properties,
                    label_columns=label_columns,
                    feature_columns=feature_columns,
                    label_shapes=label_shapes,
                    problem_types=problem_types,
                    stopping_metric=None,
                    log_metrics=None,
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
        inference_batch_size = self.config.optimization.per_device_batch_size\
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
