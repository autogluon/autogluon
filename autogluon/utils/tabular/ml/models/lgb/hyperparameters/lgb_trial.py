import logging, random
import numpy as np

from .......core import args
from ..callbacks import hpo_callback
from ....constants import BINARY, MULTICLASS, REGRESSION
from ......try_import import try_import_lightgbm

logger = logging.getLogger(__name__)  # TODO: Currently unused


@args()
def lgb_trial(args, reporter):
    """ Training script for hyperparameter evaluation of Gradient Boosting model """
    try_import_lightgbm()
    import lightgbm as lgb
    # list of args which are not model hyperparameters:
    nonparam_args = set(['directory', 'task_id', 'lgb_model', 'dataset_train_filename', 'dataset_val_filename'])
    trial_id = args.task_id # Note may not start at 0 if HPO has been run for other models with same scheduler
    directory = args.directory
    file_prefix = "trial_"+str(trial_id)+"_" # append to all file names created during this trial. Do NOT change!
    lgb_model = args.lgb_model
    lgb_model.params = lgb_model.params.copy() # ensure no remaining pointers across trials
    for key in args:
        if key not in nonparam_args:
            lgb_model.params[key] = args[key] # use these hyperparam values in this trial
    dataset_train = lgb.Dataset(directory+args.dataset_train_filename)
    dataset_val_filename = args.get('dataset_val_filename', None)
    if dataset_val_filename is not None:
        dataset_val = lgb.Dataset(directory+dataset_val_filename)
    eval_metric = lgb_model.get_eval_metric()
    if lgb_model.problem_type == BINARY:
        train_loss_name = 'binary_logloss'
    elif lgb_model.problem_type == MULTICLASS:
        train_loss_name = 'multi_logloss'
    elif lgb_model.problem_type == REGRESSION:
        train_loss_name = 'l2'
    else:
        raise ValueError("unknown problem_type for LGBModel: %s" % lgb_model.problem_type)
    lgb_model.eval_results = {}
    callbacks = []
    valid_names = ['train_set']
    valid_sets = [dataset_train]
    if dataset_val is not None:
        callbacks += [
            hpo_callback(reporter=reporter, stopping_rounds=150, metrics_to_use=[('valid_set', lgb_model.eval_metric_name)], 
                max_diff=None, ignore_dart_warning=True, verbose=False, train_loss_name=train_loss_name, eval_results=lgb_model.eval_results)
        ]
        valid_names = ['valid_set'] + valid_names
        valid_sets = [dataset_val] + valid_sets
    else:
        raise NotImplementedError("cannot call gbm hyperparameter_tune without validation dataset")
    
    num_boost_round = lgb_model.params.pop('num_boost_round', 1000)
    seed_value = lgb_model.params.pop('seed_value', None)
    train_params = {
        'params': lgb_model.params.copy(),
        'train_set': dataset_train,
        'num_boost_round': num_boost_round, 
        'valid_sets': valid_sets,
        'valid_names': valid_names,
        'evals_result': lgb_model.eval_results,
        'callbacks': callbacks,
        'verbose_eval': -1,
    }
    if type(eval_metric) != str:
        train_params['feval'] = eval_metric
    if seed_value is not None:
        train_params['seed'] = seed_value
        random.seed(seed_value)
        np.random.seed(seed_value)
    
    lgb_model.model = lgb.train(**train_params)
    lgb_model.params['num_boost_round'] = num_boost_round # re-set this value after training
    if seed_value is not None:
        lgb_model.params['seed_value'] = seed_value
    lgb_model.best_iteration = lgb_model.model.best_iteration
    # TODO: difficult to ensure these iters always match
    # if lgb_model.eval_results['best_iter'] != lgb_model.best_iteration:
    #     raise ValueError('eval_results[best_iter]=%s does not match lgb_model.best_iteration=%s' % (lgb_model.eval_results['best_iter'], lgb_model.best_iteration) )
    # print('eval_results[best_iter]=%s does not match lgb_model.best_iteration=%s' % (lgb_model.eval_results['best_iter'], lgb_model.best_iteration) )
    trial_model_file = lgb_model.save(file_prefix=file_prefix, directory=directory, return_filename=True)
    reporter(epoch=num_boost_round+1, validation_performance=lgb_model.eval_results['best_valperf'],
             train_loss=lgb_model.eval_results['best_trainloss'],
             best_iteration=lgb_model.eval_results['best_iter'],
             directory=directory, file_prefix=file_prefix, trial_model_file=trial_model_file)
    # TODO: add to reporter: time_of_trial without load/save time (isn't this just function of early-stopping point?), memory/inference ??
