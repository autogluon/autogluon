import collections, warnings, time, os, psutil, logging
from operator import gt, lt

from autogluon.core.utils.savers import save_pointer
from autogluon.core.utils.savers import save_pkl
from autogluon.core.utils import try_import_lightgbm

logger = logging.getLogger(__name__)


def save_model_callback(path, latest_model_checkpoint, interval, offset):
    def _callback(env):
        if ((env.iteration - offset) % interval == 0) & (env.iteration != 0):
            save_pkl.save(path=path, object=env.model)
            save_pointer.save(path=latest_model_checkpoint, content_path=path)
    _callback.before_iteration = True
    _callback.order = 0
    return _callback


# TODO: dart might alter previous iterations, check if this is occurring, if so then save snapshot of model when best_iteration to preserve quality
def record_evaluation_custom(path, eval_result, interval, offset=0, early_stopping_rounds=None):
    """Create a callback that records the evaluation history into ``eval_result``.

    Parameters
    ----------
    eval_result : dict
       A dictionary to store the evaluation results.

    Returns
    -------
    callback : function
        The callback that records the evaluation history into the passed dictionary.
    """
    if not isinstance(eval_result, dict):
        raise TypeError('Eval_result should be a dictionary')
    eval_result.clear()

    def _init(env):
        for data_name, _, _, _ in env.evaluation_result_list:
            eval_result.setdefault(data_name, collections.defaultdict(list))

    def _callback(env):
        if not eval_result:
            _init(env)
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            eval_result[data_name][eval_name].append(result)
        if (interval > 0) and ((env.iteration - offset) % interval == 0) and (env.iteration != 0):
            # min_error = min(eval_result['valid_set']['multi_error'])
            # print('iter:', env.iteration, 'min_error:', min_error)
            save_pkl.save(path=path, object=eval_result)
    _callback.order = 20
    return _callback


# TODO: Add option to stop if current run's metric value is X% lower, such as min 30%, current 40% -> Stop
def early_stopping_custom(stopping_rounds, first_metric_only=False, metrics_to_use=None, start_time=None, time_limit=None, verbose=True, max_diff=None, ignore_dart_warning=False, manual_stop_file=None, train_loss_name=None, reporter=None):
    """Create a callback that activates early stopping.

    Note
    ----
    Activates early stopping.
    The model will train until the validation score stops improving.
    Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.
    To check only the first metric set ``first_metric_only`` to True.

    Parameters
    ----------
    stopping_rounds : int
       The possible number of rounds without the trend occurrence.
    first_metric_only : bool, optional (default=False)
       Whether to use only the first metric for early stopping.
    verbose : bool, optional (default=True)
        Whether to print message with early stopping information.
    train_loss_name : str, optional (default=None):
        Name of metric that contains training loss value.
    reporter : optional (default=None):
        reporter object from AutoGluon scheduler.

    Returns
    -------
    callback : function
        The callback that activates early stopping.
    """
    best_score = []
    best_iter = []
    best_score_list = []
    best_trainloss = []  # stores training losses at corresponding best_iter
    cmp_op = []
    enabled = [True]
    indices_to_check = []
    timex = [time.time()]
    mem_status = psutil.Process()
    init_mem_rss = []
    init_mem_avail = []

    def _init(env):
        if not ignore_dart_warning:
            enabled[0] = not any((boost_alias in env.params
                                  and env.params[boost_alias] == 'dart') for boost_alias in ('boosting',
                                                                                             'boosting_type',
                                                                                             'boost'))
        if not enabled[0]:
            warnings.warn('Early stopping is not available in dart mode')
            return
        if not env.evaluation_result_list:
            raise ValueError('For early stopping, '
                             'at least one dataset and eval metric is required for evaluation')

        if verbose:
            msg = "Training until validation scores don't improve for {} rounds."
            logger.debug(msg.format(stopping_rounds))
            if manual_stop_file:
                logger.debug('Manually stop training by creating file at location: ', manual_stop_file)

        for eval_ret in env.evaluation_result_list:
            best_iter.append(0)
            best_score_list.append(None)
            best_trainloss.append(None)
            if eval_ret[3]:
                best_score.append(float('-inf'))
                cmp_op.append(gt)
            else:
                best_score.append(float('inf'))
                cmp_op.append(lt)

        if metrics_to_use is None:
            for i in range(len(env.evaluation_result_list)):
                indices_to_check.append(i)
                if first_metric_only:
                    break
        else:
            for i, eval in enumerate(env.evaluation_result_list):
                if (eval[0], eval[1]) in metrics_to_use:
                    indices_to_check.append(i)
                    if first_metric_only:
                        break

        init_mem_rss.append(mem_status.memory_info().rss)
        init_mem_avail.append(psutil.virtual_memory().available)

    def _callback(env):
        try_import_lightgbm()
        from lightgbm.callback import _format_eval_result, EarlyStopException
        if not cmp_op:
            _init(env)
        if not enabled[0]:
            return
        if train_loss_name is not None:
            train_loss_evals = [eval for eval in env.evaluation_result_list if eval[0] == 'train_set' and eval[1] == train_loss_name]
            train_loss_val = train_loss_evals[0][2]
        else:
            train_loss_val = 0.0
        for i in indices_to_check:
            eval_result = env.evaluation_result_list[i]
            _, eval_metric, score, greater_is_better = eval_result
            if best_score_list[i] is None or cmp_op[i](score, best_score[i]):
                best_score[i] = score
                best_iter[i] = env.iteration
                best_score_list[i] = env.evaluation_result_list
                best_trainloss[i] = train_loss_val
            if reporter is not None:  # Report current best scores for iteration, used in HPO
                if i == indices_to_check[0]:  # TODO: documentation needs to note that we assume 0th index is the 'official' validation performance metric.
                    if cmp_op[i] == gt:
                        validation_perf = score
                    else:
                        validation_perf = -score
                    reporter(epoch=env.iteration + 1,
                             validation_performance=validation_perf,
                             train_loss=best_trainloss[i],
                             best_iter_sofar=best_iter[i] + 1,
                             best_valperf_sofar=best_score[i],
                             eval_metric=eval_metric,
                             greater_is_better=greater_is_better,
                             )
            if env.iteration - best_iter[i] >= stopping_rounds:
                if verbose:
                    logger.log(15, 'Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            elif (max_diff is not None) and (abs(score - best_score[i]) > max_diff):
                if verbose:
                    logger.debug('max_diff breached!')
                    logger.debug(abs(score - best_score[i]))
                    logger.log(15, 'Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if env.iteration == env.end_iteration - 1:
                if verbose:
                    logger.log(15, 'Did not meet early stopping criterion. Best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if verbose:
                logger.debug((env.iteration - best_iter[i], eval_result))
        if manual_stop_file:
            if os.path.exists(manual_stop_file):
                i = indices_to_check[0]
                logger.log(20, 'Found manual stop file, early stopping. Best iteration is:\n[%d]\t%s' % (
                    best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
        if time_limit:
            time_elapsed = time.time() - start_time
            time_left = time_limit - time_elapsed
            if time_left <= 0:
                i = indices_to_check[0]
                logger.log(20, '\tRan out of time, early stopping on iteration ' + str(env.iteration+1) + '. Best iteration is:\n\t[%d]\t%s' % (
                    best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])

        # TODO: Add toggle parameter to early_stopping to disable this
        # TODO: Identify optimal threshold values for early_stopping based on lack of memory
        if env.iteration % 10 == 0:
            available = psutil.virtual_memory().available
            cur_rss = mem_status.memory_info().rss

            if cur_rss < init_mem_rss[0]:
                init_mem_rss[0] = cur_rss
            estimated_model_size_mb = (cur_rss - init_mem_rss[0]) >> 20
            available_mb = available >> 20

            model_size_memory_ratio = estimated_model_size_mb / available_mb
            if verbose or (model_size_memory_ratio > 0.25):
                logging.debug('Available Memory: '+str(available_mb)+' MB')
                logging.debug('Estimated Model Size: '+str(estimated_model_size_mb)+' MB')

            early_stop = False
            if model_size_memory_ratio > 1.0:
                logger.warning('Warning: Large GBM model size may cause OOM error if training continues')
                logger.warning('Available Memory: '+str(available_mb)+' MB')
                logger.warning('Estimated GBM model size: '+str(estimated_model_size_mb)+' MB')
                early_stop = True

            # TODO: We will want to track size of model as well, even if we early stop before OOM, we will still crash when saving if the model is large enough
            if available_mb < 512:  # Less than 500 MB
                logger.warning('Warning: Low available memory may cause OOM error if training continues')
                logger.warning('Available Memory: '+str(available_mb)+' MB')
                logger.warning('Estimated GBM model size: '+str(estimated_model_size_mb)+' MB')
                early_stop = True

            if early_stop:
                logger.warning('Warning: Early stopped GBM model prior to optimal result to avoid OOM error. Please increase available memory to avoid subpar model quality.')
                logger.log(15, 'Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_iter[0] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[0]])))
                raise EarlyStopException(best_iter[0], best_score_list[0])

    _callback.order = 30
    return _callback
