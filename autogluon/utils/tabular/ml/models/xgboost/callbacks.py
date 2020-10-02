import time
import psutil
from xgboost import rabit
from xgboost.core import EarlyStopException


def _fmt_metric(value, show_stdv=True):
    """format metric string"""
    if len(value) == 2:
        return '{0}:{1:.5f}'.format(value[0], value[1])
    if len(value) == 3:
        if show_stdv:
            return  '{0}:{1:.5f}+{2:.5f}'.format(value[0], value[1], value[2])
        return '{0}:{1:.5f}'.format(value[0], value[1])
    raise ValueError("wrong metric value")


def early_stop_custom(stopping_rounds, start_time=None, time_limit=None, maximize=False, verbose=True):
    """Create a callback that activates early stoppping.
    Validation error needs to decrease at least
    every **stopping_rounds** round(s) to continue training.
    Requires at least one item in **evals**.
    If there's more than one, will use the last.
    Returns the model from the last iteration (not the best one).
    If early stopping occurs, the model will have three additional fields:
    ``bst.best_score``, ``bst.best_iteration`` and ``bst.best_ntree_limit``.
    (Use ``bst.best_ntree_limit`` to get the correct value if ``num_parallel_tree``
    and/or ``num_class`` appears in the parameters)
    Parameters
    ----------
    stopping_rounds : int
       The stopping rounds before the trend occur.
    start_time : float
        start time of model training.
    time_limit : float
        limited time to train model.
    maximize : bool
        Whether to maximize evaluation metric.
    verbose : optional, bool
        Whether to print message about early stopping information.
    Returns
    -------
    callback : function
        The requested callback function.
    """
    state = {}
    mem_status = psutil.Process()


    def init(env):
        """internal function"""
        bst = env.model

        state['init_mem_rss'] = mem_status.memory_info().rss
        available = psutil.virtual_memory().available

        if not env.evaluation_result_list:
            raise ValueError('For early stopping you need at least one set in evals.')
        if len(env.evaluation_result_list) > 1 and verbose:
            msg = ("Multiple eval metrics have been passed: "
                   "'{0}' will be used for early stopping.\n\n")
            rabit.tracker_print(msg.format(env.evaluation_result_list[-1][0]))
        maximize_metrics = ('auc', 'aucpr', 'map', 'ndcg')
        maximize_at_n_metrics = ('auc@', 'aucpr@', 'map@', 'ndcg@')
        maximize_score = maximize
        metric_label = env.evaluation_result_list[-1][0]  # TODO: default function only considers last validation performance metric.
        metric = metric_label.split('-', 1)[-1]

        if any(metric.startswith(x) for x in maximize_at_n_metrics):
            maximize_score = True

        if any(metric.split(":")[0] == x for x in maximize_metrics):
            maximize_score = True

        if verbose and env.rank == 0:
            msg = "Will train until {} hasn't improved in {} rounds.\n"
            rabit.tracker_print(msg.format(metric_label, stopping_rounds))

        state['maximize_score'] = maximize_score
        state['best_iteration'] = 0
        if maximize_score:
            state['best_score'] = float('-inf')
        else:
            state['best_score'] = float('inf')
        msg = '[%d]\t%s' % (
            env.iteration,
            '\t'.join([_fmt_metric(x) for x in env.evaluation_result_list]))
        state['best_msg'] = msg

        if bst is not None:
            if bst.attr('best_score') is not None:
                state['best_score'] = float(bst.attr('best_score'))
                state['best_iteration'] = int(bst.attr('best_iteration'))
                state['best_msg'] = bst.attr('best_msg')
            else:
                bst.set_attr(best_iteration=str(state['best_iteration']))
                bst.set_attr(best_score=str(state['best_score']))
        else:
            assert env.cvfolds is not None

    def callback(env):
        """internal function"""
        if not state:
            init(env)
        score = env.evaluation_result_list[-1][1]
        best_score = state['best_score']
        best_iteration = state['best_iteration']
        maximize_score = state['maximize_score']
        
        if (maximize_score and score > best_score) or \
                (not maximize_score and score < best_score):
            msg = '[%d]\t%s' % (
                env.iteration,
                '\t'.join([_fmt_metric(x) for x in env.evaluation_result_list]))
            state['best_msg'] = msg
            state['best_score'] = score
            state['best_iteration'] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(best_score=str(state['best_score']),
                                   best_iteration=str(state['best_iteration']),
                                   best_msg=state['best_msg'])
        elif env.iteration - best_iteration >= stopping_rounds:
            best_msg = state['best_msg']
            if verbose and env.rank == 0:
                msg = "Stopping. Best iteration:\n{}\n\n"
                rabit.tracker_print(msg.format(best_msg))
            raise EarlyStopException(best_iteration)

        if env.iteration % 10 == 0:
            available = psutil.virtual_memory().available
            cur_rss = mem_status.memory_info().rss
            # rabit.tracker_print(f"{state}\n")
            # rabit.tracker_print(f"CUR_RSS: {cur_rss}\n")
            if cur_rss < state['init_mem_rss']:
                state['init_mem_rss'] = cur_rss
            estimated_model_size_mb = (cur_rss - state['init_mem_rss']) >> 20
            available_mb = available >> 20

            model_size_memory_ratio = estimated_model_size_mb / available_mb
            if verbose and (model_size_memory_ratio > 0.25):
                rabit.tracker_print(f'Available Memory: {available_mb} MB\n')
                rabit.tracker_print(f'Estimated XGB model size: {estimated_model_size_mb} MB\n')

            early_stop = False
            if (model_size_memory_ratio > 1.0) or (available_mb < 512):
                rabit.tracker_print('Warning: Large XGB model size may cause OOM error if training continues\n')
                rabit.tracker_print(f'Available Memory: {available_mb} MB\n')
                rabit.tracker_print(f'Estimated XGB model size: {estimated_model_size_mb} MB\n')
                early_stop = True
            
            if early_stop:
                if verbose and env.rank == 0:
                    rabit.tracker_print(f'Warning: Early stopped GBM model prior to optimal result to avoid OOM error. Please increase available memory to avoid subpar model quality.\n')
                    rabit.tracker_print(f'Early stopping. best iteration is:\n[{env.iteration}]\t{best_score}')
                raise EarlyStopException(best_iteration)
        
        if time_limit:
            time_elapsed = time.time() - start_time
            time_left = time_limit - time_elapsed
            if time_left <= 0:
                if verbose and env.rank == 0:
                    rabit.tracker_print(f"Ran out of time, early stopping on iteration {env.iteration}. Best iteration is:\n\t[{best_iteration}]\t{best_score}")
                    rabit.tracker_print(state['best_msg'])
                raise EarlyStopException(best_iteration)

    return callback
