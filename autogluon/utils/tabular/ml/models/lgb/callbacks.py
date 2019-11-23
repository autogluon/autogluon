from autogluon.utils.tabular.utils.savers import save_pkl, save_pointer
import collections
import warnings
from operator import gt, lt
from lightgbm.compat import range_
import time
import os
from lightgbm.callback import _format_eval_result, EarlyStopException
import psutil


# callback
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
def early_stopping_custom(stopping_rounds, first_metric_only=False, metrics_to_use=None, verbose=True, max_diff=None, ignore_dart_warning=False, manual_stop_file=None):
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

    Returns
    -------
    callback : function
        The callback that activates early stopping.
    """
    best_score = []
    best_iter = []
    best_score_list = []
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
            print(msg.format(stopping_rounds))
            if manual_stop_file:
                print('Manually stop training by creating file at location: ', manual_stop_file)

        for eval_ret in env.evaluation_result_list:
            best_iter.append(0)
            best_score_list.append(None)
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
        if not cmp_op:
            _init(env)
        if not enabled[0]:
            return

        for i in indices_to_check:
            score = env.evaluation_result_list[i][2]
            if best_score_list[i] is None or cmp_op[i](score, best_score[i]):
                best_score[i] = score
                best_iter[i] = env.iteration
                best_score_list[i] = env.evaluation_result_list
            elif env.iteration - best_iter[i] >= stopping_rounds:
                if verbose:
                    print('Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            elif (max_diff is not None) and (abs(score - best_score[i]) > max_diff):
                if verbose:
                    print('max_diff breached!')
                    print(abs(score - best_score[i]))
                    print('Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if env.iteration == env.end_iteration - 1:
                if verbose:
                    print('Did not meet early stopping. Best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if verbose:
                print(env.iteration - best_iter[i], env.evaluation_result_list[i])
        if manual_stop_file:
            if os.path.exists(manual_stop_file):
                i = indices_to_check[0]
                print('Found manual stop file, early stopping. Best iteration is:\n[%d]\t%s' % (
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
                print('Available Memory:', available_mb, 'MB')
                print('Estimated Model Size:', estimated_model_size_mb, 'MB')

            early_stop = False
            if model_size_memory_ratio > 1.0:
                print('Warning: Large model size may cause OOM error during saving if training continues')
                print('Available Memory:', available_mb, 'MB')
                print('Estimated Model Size:', estimated_model_size_mb, 'MB')
                early_stop = True

            # TODO: We will want to track size of model as well, even if we early stop before OOM, we will still crash when saving if the model is large enough
            if available_mb < 512:  # Less than 500 MB
                print('Warning: Low available memory may cause OOM error if training continues')
                print('Available Memory:', available_mb, 'MB')
                print('Estimated Model Size:', estimated_model_size_mb, 'MB')
                early_stop = True

            if early_stop:
                print('Warning: Early stopping prior to optimal result to avoid OOM error, increase memory allocation to maximize model quality.')
                print('Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_iter[0] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[0]])))
                raise EarlyStopException(best_iter[0], best_score_list[0])

    _callback.order = 30
    return _callback


def hpo_callback(reporter, stopping_rounds, first_metric_only=False, metrics_to_use=None, 
                 verbose=True, max_diff=None, ignore_dart_warning=False, manual_stop_file=None, 
                 train_loss_name=None, eval_results={}):
    """Create a callback that activates early stopping to use in HPO.

    Note
    ----
    metrics_to_use must be list of len == 1.  We do NOT support multiple metrics_to_use here!
    The model will train until the validation score stops improving.
    Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.
    To check only the first metric set ``first_metric_only`` to True.

    Parameters
    ----------
    reporter: reporter object from AutoGluon scheduler
    stopping_rounds : int
       The possible number of rounds without the trend occurrence.
    first_metric_only : bool, optional (default=False)
       Whether to use only the first metric for early stopping.
    verbose : bool, optional (default=True)
        Whether to print message with early stopping information
    train_loss_name (str): Name of metric that contains training loss value
    eval_results (dict): Where to store stuff
    
    Returns
    -------
    callback : function
        The callback that activates early stopping
    """
    best_score = []
    best_iter = []
    best_score_list = []
    best_trainloss = [] # stores training losses at corresponding best_iter
    cmp_op = []
    indices_to_check = []
    enabled = [True]
    timex = [time.time()]
    if len(metrics_to_use) != 1:
        raise ValueError("hpo_callback() can only be used when metrics_to_use is list of length == 1")

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
            print(msg.format(stopping_rounds))
            if manual_stop_file:
                print('Manually stop training by creating file at location: ', manual_stop_file)
        
        for eval_ret in env.evaluation_result_list:
            best_iter.append(0)
            best_score_list.append(None)
            best_trainloss.append(0.0)
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

    def _callback(env):
        cur_time = time.time()
        # if verbose:
        #     print(cur_time - timex[0])
        timex[0] = cur_time
        if not cmp_op:
            _init(env)
        if not enabled[0]:
            return
        if train_loss_name is not None:
            train_loss_evals = [eval for i, eval in enumerate(env.evaluation_result_list) if eval[0]=='train_set' and eval[1]==train_loss_name]
            train_loss_val = train_loss_evals[0][2]
        else:
            train_loss_val = 0.0
        for i in indices_to_check:
            score = env.evaluation_result_list[i][2]
            if i == indices_to_check[0]: # TODO: documentation needs to note that we assume 0th index is the 'official' validation performance metric.
                if cmp_op[i] == gt:
                    validation_perf = score
                else:
                    validation_perf = -score
            if best_score_list[i] is None or cmp_op[i](score, best_score[i]):
                best_score[i] = score
                best_iter[i] = env.iteration
                best_score_list[i] = env.evaluation_result_list
                if train_loss_name is not None:
                    best_trainloss[i] = train_loss_val # same for all i
            elif env.iteration - best_iter[i] >= stopping_rounds:
                if verbose:
                    print('Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            elif (max_diff is not None) and (abs(score - best_score[i]) > max_diff):
                if verbose:
                    print('max_diff breached!')
                    print(abs(score - best_score[i]))
                    print('Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if env.iteration == env.end_iteration - 1:
                if verbose:
                    print('Did not meet early stopping. Best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if verbose:
                print(env.iteration - best_iter[i], env.evaluation_result_list[i])
            if manual_stop_file:
                if os.path.exists(manual_stop_file):
                    print('Found manual stop file, early stopping. Best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                    raise EarlyStopException(best_iter[i], best_score_list[i])
        # TODO: This should be moved inside for loop at the start, otherwise it won't record the final iteration
        idx = indices_to_check[0]
        eval_results['best_iter'] = best_iter[idx] + 1 # add one to index to align with lightgbm best_iteration instance variable.
        eval_results['best_valperf'] = best_score[idx] # validation performance at round = best_iter
        eval_results['best_trainloss'] = best_trainloss[idx] # training loss at round = best_iter
        reporter(epoch=env.iteration, validation_performance=validation_perf, train_loss=train_loss_val, 
                 best_iter_sofar=eval_results['best_iter'], best_valperf_sofar=eval_results['best_valperf'])
        # TODO: Add memory checks as in early_stopping_custom
    _callback.order = 30
    return _callback



