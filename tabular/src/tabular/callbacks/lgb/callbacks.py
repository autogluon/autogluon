from tabular.utils.savers import save_pkl, save_pointer
import collections
import warnings
from operator import gt, lt
from lightgbm.compat import range_
import time
import os
from lightgbm.callback import _format_eval_result, EarlyStopException


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
    timex = [time.time()]

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

        print(env.evaluation_result_list)
        for eval_ret in env.evaluation_result_list:
            best_iter.append(0)
            best_score_list.append(None)
            if eval_ret[3]:
                best_score.append(float('-inf'))
                cmp_op.append(gt)
            else:
                best_score.append(float('inf'))
                cmp_op.append(lt)

    def _callback(env):
        cur_time = time.time()
        # if verbose:
        #     print(cur_time - timex[0])
        timex[0] = cur_time
        if not cmp_op:
            _init(env)
        if not enabled[0]:
            return
        if metrics_to_use is None:
            indices_to_check = range(len(env.evaluation_result_list))
        else:
            indices_to_check = [i for i, eval in enumerate(env.evaluation_result_list) if (eval[0], eval[1]) in metrics_to_use]
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
                    print('Found manual stop file, early stopping. Best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                    raise EarlyStopException(best_iter[i], best_score_list[i])
            if first_metric_only:  # the only first metric is used for early stopping
                break

    _callback.order = 30
    return _callback
