import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import boto3
from sklearn import metrics
import os
import logging
import tempfile

from .files import mkdir
from .serialization import save

logger = logging.getLogger(__name__)

__all__ = ['make_results_path',
           'make_training_history_callback',
           'default_num_init_random',
           'analyze_results',
           'compute_experiment_scores',
           'load_results',
           'load_result']


TOGGLE_PREFIX = 'toggle_'


# Supported SageMaker instance types, together with number of vCPUs
sagemaker_instances = {
    'ml.t2.medium': 2,
    'ml.t2.large': 2,
    'ml.t2.xlarge': 4,
    'ml.t2.2xlarge': 8,
    'ml.t3.medium': 2,
    'ml.t3.large': 2,
    'ml.t3.xlarge': 4,
    'ml.t3.2xlarge': 8,
    'ml.m5.large': 2,
    'ml.m5.xlarge': 4,
    'ml.m5.2xlarge': 8,
    'ml.m5.4xlarge': 16,
    'ml.m5.12xlarge': 48,
    'ml.m5.24xlarge': 96,
    'ml.m4.xlarge': 4,
    'ml.m4.4xlarge': 16,
    'ml.m4.10xlarge': 40,
    'ml.m4.16xlarge': 64,
    'ml.m5d.large': 2,
    'ml.m5d.xlarge': 4,
    'ml.m5d.2xlarge': 8,
    'ml.m5d.4xlarge': 16,
    'ml.m5d.8xlarge': 32,
    'ml.m5d.12xlarge': 48,
    'ml.m5d.24xlarge': 96,
    'ml.r5.large': 2,
    'ml.r5.xlarge': 4,
    'ml.r5.2xlarge': 8,
    'ml.r5.4xlarge': 16,
    'ml.r5.12xlarge': 48,
    'ml.r5.24xlarge': 96,
    'ml.r5d.large': 2,
    'ml.r5d.xlarge': 4,
    'ml.r5d.2xlarge': 8,
    'ml.r5d.4xlarge': 16,
    'ml.r5d.8xlarge': 32,
    'ml.r5d.12xlarge': 48,
    'ml.r5d.16xlarge': 64,
    'ml.r5d.24xlarge': 96,
    'ml.c5.large': 2,
    'ml.c5.xlarge': 4,
    'ml.c5.2xlarge': 8,
    'ml.c5.4xlarge': 16,
    'ml.c5.9xlarge': 36,
    'ml.c5.18xlarge': 72,
    'ml.c5d.xlarge': 4,
    'ml.c5d.2xlarge': 8,
    'ml.c5d.4xlarge': 16,
    'ml.c5d.9xlarge': 36,
    'ml.c5d.18xlarge': 72,
    'ml.c4.large': 2,
    'ml.c4.xlarge': 4,
    'ml.c4.2xlarge': 8,
    'ml.c4.4xlarge': 16,
    'ml.c4.8xlarge': 36,
    'ml.p3.2xlarge': 8,
    'ml.p3.8xlarge': 32,
    'ml.p3.16xlarge': 64,
    'ml.p3dn.24xlarge': 96,
    'ml.p2.xlarge': 4,
    'ml.p2.8xlarge': 32,
    'ml.p2.16xlarge': 64,
    'ml.g4dn.xlarge': 4,
    'ml.g4dn.2xlarge': 8,
    'ml.g4dn.4xlarge': 16,
    'ml.g4dn.8xlarge': 32,
    'ml.g4dn.12xlarge': 48,
    'ml.g4dn.16xlarge': 64
}


def _upload_file(file_name, bucket=None, object_name=None):
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name
    # Upload the file
    s3_client = boto3.client('s3')
    s3_client.upload_file(file_name, bucket, object_name)


def _process_training_history(
        training_history, config_history, start_timestamp, runtime_fn,
        error_fn, resource_attr):
    task_dfs = []

    for task_id in training_history:
        task_list = training_history[task_id]
        if task_id in config_history:
            config_dct = config_history[task_id]
            # Makes sure that if a config_dct key is already in task_dct,
            # the corresponding entry is not changed
            task_list = [
                dict(config_dct, **task_dct) for task_dct in task_list]
        task_df = pd.DataFrame(task_list)
        task_df = task_df.assign(
            task_id=task_id,
            runtime=runtime_fn(task_df, start_timestamp),
            error=error_fn(task_df),
            target_epoch=task_df[resource_attr].iloc[-1])
        # TODO: support other summarization methods, such as min/max.
        # eval_summary = eval_df.iloc[-1]
        # Assign name so we get trial numbers
        # eval_summary.name = eval_index
        task_dfs.append(task_df)

    result = pd.concat(task_dfs, axis="index", ignore_index=True, sort=True)

    # re-order by runtime
    result = result.sort_values(by="runtime")

    # calculate incumbent best -- the cumulative minimum of the error.
    result = result.assign(best=result["error"].cummin())

    return result


def _path_experiment(params):
    benchmark_name = params['benchmark_name']
    dataset_name = params.get('dataset_name')
    if dataset_name is not None:
        benchmark_name = '_'.join((benchmark_name, dataset_name))
    speedup_factor = params.get('speedup_factor')
    if speedup_factor is not None:
        benchmark_name += '_speedup{}'.format(speedup_factor)
    return benchmark_name


def _path_instance_resource(params):
    instance_count = int(params.get('instance_count', 1))
    name = params['instance_type']
    if instance_count > 1:
        name += '_cnt{}'.format(instance_count)
    name += '_cpus{}_gpus{}'.format(params['num_cpus'], params['num_gpus'])
    return name


key_abbreviations = {
    'searcher_num_init_random': 'initrandom',
    'searcher_num_init_candidates': 'initcand',
    'searcher_num_fantasy_samples': 'fantasies',
    'searcher_resource_acq': 'resacq',
    'searcher_resource_acq_bohb_threshold': 'resacqthres',
    'searcher_resource_kernel': 'kernel'
}


def make_results_path(
        params, base_path, mkdir=True, abbrev_keys=True):
    exclude_keys = {
        's3folder', 's3bucket', 'instance_type', 'n_runs', 'logs', '_exp_id',
        'hostfile', 'write_s3', 'store_results_period', 'debug_log',
        'num_trials', 'scheduler_timeout', 'benchmark_name', 'dataset_name',
        'scheduler', 'searcher', 'run_id', 'account', 'expname', 'mode',
        'instance_count', 'timeout', 'grace', 'region', 'role', 'name',
        'num_cpus', 'num_gpus', 'dataset_path', 'speedup_factor',
        'ignore_load_time', 'checkpoint_fname', 'start_run_id', 'end_run_id',
        'searcher_profiling', 'store_results_suffix_period'}
    filtered_params = dict()
    filtered_params_orig = dict()
    for k, v in params.items():
        if (v is not None) and (k not in exclude_keys) and \
            (not k.startswith(TOGGLE_PREFIX)):
            filtered_params_orig[k] = v
            if abbrev_keys:
                k_abbrev = key_abbreviations.get(k, k)
                filtered_params[k_abbrev] = v
    if not abbrev_keys:
        filtered_params = filtered_params_orig

    benchmark_name = _path_experiment(params)
    instance_resource = _path_instance_resource(params)
    paths = [benchmark_name, instance_resource, params['scheduler'],
             params['searcher']]
    if filtered_params:
        # Final directory name made of all remaining key-values
        # ATTENTION: This may be longer than the limit for directory
        # names. We use 'key_abbreviations' to at least shorten the
        # keys
        parts = []
        for k in sorted(filtered_params.keys()):
            parts.extend([k, str(filtered_params[k])])
        paths.append('_'.join(parts))
    results_path = Path(base_path).joinpath(*paths)

    if mkdir:
        results_path.mkdir(parents=True, exist_ok=True)

    for key in (
            'instance_type', 'searcher', 'scheduler', 'num_trials',
            'scheduler_timeout', 'benchmark_name', 'dataset_name',
            'num_cpus', 'num_gpus'):
        if key in params:
            filtered_params_orig[key] = params[key]

    return results_path, filtered_params_orig


def _extract_total_train_time(
        training_history, resource_attr, elapsed_time_attr):
    sum_res = 0
    sum_time = 0
    for reports in training_history.values():
        res_times = [(int(x[resource_attr]), float(x[elapsed_time_attr]))
                     for x in reports]
        max_res, max_time = max(res_times, key=lambda x: x[0])
        sum_res += max_res
        sum_time += max_time
    return sum_time, sum_res


def make_training_history_callback(
        params, runtime_fn, error_fn, resource_attr, elapsed_time_attr):
    """
    Generates training_history callback function, which stores training_history.
    The callback function is called at certain intervals.

    :param params:
    :param runtime_fn: Function runtime_fn(df, start_timestamp). df is the
        dataframe for one task, containing all reported records. This
        function returns a series for the runtime (wall clock time) from the
        start of the experiment (start_timestamp).
        The default uses the 'time_step' entry, which is a timestamp of the
        train_fn when a record is reported.
    :param error_fn: Function error_fn(df). df is the
        dataframe for one task, containing all reported records. This function
        returns a series of error metric values (smaller is better). Note that
        internally, AutoGluon uses reward (larger is better).
    :param resource_attr: Resource attribute (e.g., 'epoch')
    :return: training_history_callback, function with arguments training_history
        and start_timestamp
    """
    local_root = tempfile.mkdtemp()
    results_path, meta_dict = make_results_path(
        params, base_path=params['s3folder'])
    trainhist_fname = str(results_path / "run_{}.csv".format(
        int(params['run_id'])))
    trainhist_fname_local = os.path.join(local_root, trainhist_fname)
    logger.info("Training history written to {}".format(
        trainhist_fname_local))
    path = os.path.dirname(trainhist_fname_local)
    try:
        mkdir(path)
    except Exception:
        logger.info(
            "[!!!] Failed to create local path {}".format(path))
    write_s3 = params['write_s3']
    s3bucket = params['s3bucket']
    checkpoint_fname = params.get('checkpoint_fname')
    checkpoint_fname_local = None
    if checkpoint_fname is not None:
        checkpoint_fname = os.path.join(params['s3folder'], checkpoint_fname)
        checkpoint_fname_local = os.path.join(local_root, checkpoint_fname)
        logger.info("Scheduler checkpoint written to {}".format(
            checkpoint_fname_local))
        path = os.path.dirname(checkpoint_fname_local)
        try:
            mkdir(path)
        except Exception:
            logger.info(
                "[!!!] Failed to create local path {}".format(path))
    store_results_suffix_period = params.get('store_results_suffix_period')
    if store_results_suffix_period is not None:
        assert store_results_suffix_period >= 2, \
            "'store_results_suffix_period' must be integer >= 2"
    callback_num_calls = [0]  # Need something mutable
    # TODO: Also write out meta_dict as JSON

    def training_history_callback(
            training_history, start_timestamp, config_history=None,
            state_dict=None):
        # Process training_history, store locally
        logger.info("Entering training_history_callback")
        training_history = training_history.copy()
        if config_history is None:
            config_history = dict()
        results_df = _process_training_history(
            training_history, config_history, start_timestamp=start_timestamp,
            runtime_fn=runtime_fn, error_fn=error_fn,
            resource_attr=resource_attr)
        _trainhist_fname_local = trainhist_fname_local
        _trainhist_fname = trainhist_fname
        num_calls = callback_num_calls[0]
        if store_results_suffix_period is not None:
            suffix = '.{}'.format(num_calls % store_results_suffix_period)
            _trainhist_fname_local += suffix
            _trainhist_fname += suffix
        callback_num_calls[0] = num_calls + 1
        sum_time, sum_res = _extract_total_train_time(
            training_history, resource_attr, elapsed_time_attr)
        msg = "Total work done so far on all workers: training_time {:.2f} secs, {} epochs".format(
            sum_time, sum_res)
        logger.info(msg)
        msg = "Writing training_history ({} rows) to {}".format(
            results_df.shape[0], _trainhist_fname_local)
        results_df.to_csv(_trainhist_fname_local)
        _meta_fname_local = _trainhist_fname_local + ".meta"
        with open(_meta_fname_local, 'w') as f:
            f.write("sum_epochs, sum_traintime\n")
            f.write ("{}, {:.2f}\n".format(sum_res, sum_time))
        if write_s3:
            # Copy file to S3
            msg += " and upload to S3"
            _upload_file(_trainhist_fname_local, bucket=s3bucket,
                         object_name=_trainhist_fname)
            _upload_file(_meta_fname_local, bucket=s3bucket,
                         object_name=_trainhist_fname + '.meta')
        logger.info(msg)
        if checkpoint_fname is not None:
            if state_dict is not None:
                try:
                    msg = "Writing checkpoint to {}".format(
                        checkpoint_fname_local)
                    save(state_dict, checkpoint_fname_local)
                    if write_s3:
                        msg += " and upload to S3"
                        _upload_file(
                            checkpoint_fname_local, bucket=s3bucket,
                            object_name=checkpoint_fname)
                        logger.info(msg)
                except Exception:
                    logger.info("[!!!] training_history_callback: Could not store scheduler checkpoint to {} or upload to S3".format(
                        checkpoint_fname_local))
            else:
                logger.info(
                    "[!!!] training_history_callback: checkpoint_fname is "
                    "given, but scheduler does not pass state_dict")

    return training_history_callback


def default_num_init_random(params):
    """
    Default value for num_init_random. This is set to the number of workers
    plus two. If instance_count > 1, we assume each instance runs one worker,
    which is the case for GPU instances.
    If instance_count == 1 and no GPU is used, the number of workers is the
    number of vCPUs divided by num_cpus.

    :param params:
    :return: Default value for num_init_random, or None if conditions do not
        apply (falls back to overall default value then)
    """
    instance_type = params['instance_type']
    instance_count = int(params['instance_count'])
    num_gpus = int(params['num_gpus'])
    def_val = None
    if instance_count == 1:
        if num_gpus == 0 and instance_type in sagemaker_instances:
            # Default: num_workers + 2
            num_total_cpus = sagemaker_instances[instance_type]
            def_val = (num_total_cpus // int(params['num_cpus'])) + 2
    else:
        # Assume: If instance_count > 1, workers do not run jobs in parallel
        def_val = instance_count + 2
    return def_val


def _fill_trajectory(performance_list, time_list, replace_nan=np.NaN):
    frame_dict = OrderedDict()
    for c, (p, t) in enumerate(zip(performance_list, time_list)):
        assert len(p) == len(t), \
            "({}) Array length mismatch: {} != {}".format(c, len(p), len(t))
        ds = pd.Series(data=p, index=t)
        ds = ds[~ds.index.duplicated(keep='first')]
        frame_dict[str(c)] = ds
    merged = pd.DataFrame(frame_dict)
    merged = merged.ffill()
    performance = merged.to_numpy()
    time_ = merged.index.values
    performance[np.isnan(performance)] = replace_nan
    assert np.isfinite(performance).all(), \
        "\nCould not merge lists, because \n"\
        "\t(a) one list is empty?\n"\
        "\t(b) the lists do not start with the same times and"\
        " replace_nan is not set?\n"\
        "\t(c) any other reason."
    return performance, time_


def _prepare_errors_time(errors, runtimes, replace_nan=None):
    if replace_nan is None:
        replace_nan = 0.5
    start_t = np.max([runtimes[i][0] for i in range(len(runtimes))])
    te, time = _fill_trajectory(errors, runtimes, replace_nan=replace_nan)
    idx = time.tolist().index(start_t)
    te = te[idx:, :]
    time = time[idx:].reshape((-1,))
    return te, time, start_t


def _average_performance_over_time(errors, runtimes, replace_nan=None):
    te, time, _ = _prepare_errors_time(
        errors, runtimes, replace_nan=replace_nan)
    mean = np.mean(te.T, axis=0)
    std = np.std(te.T, axis=0)
    return time, mean, std


def _median_performance_over_time(errors, runtimes, replace_nan=None):
    te, time, _ = _prepare_errors_time(
        errors, runtimes, replace_nan=replace_nan)
    median = np.median(te.T, axis=0)
    return time, median


def _percentiles_performance_over_time(
        errors, runtimes, q=[25, 50, 75], replace_nan=None):
    te, time, _ = _prepare_errors_time(
        errors, runtimes, replace_nan=replace_nan)
    percs = np.percentile(te, q=q, axis=1)
    return time, percs


def compute_experiment_scores(results: dict, max_error=0.5):
    """
    results[k]['runtime'], results[k]['error'] provide wall-clock times and
    error values for an experiment indexed by k, they are lists of n_runs
    vectors (entries for each of n_runs repetitions). We compute normalized
    AUC and minimum error value attained, appending these as results[k]['auc']
    and results[k]['minerror']. These are vectors of size n_runs.

    :param results:
    :return:
    """
    for name in results.keys():
        result = results[name]
        num_entries = [len(x) for x in result['error']]
        errors, runtime, start_t = _prepare_errors_time(
            result['error'], result['runtime'], replace_nan=max_error)
        assert errors.ndim == 2 and errors.shape[0] == runtime.size
        aucs = []
        minerrors = []
        denom = runtime[-1]
        assert denom > start_t, \
            "UUPS! name = {}: denom = {}, start_t = {}".format(
                name, denom, start_t)
        for run_id in range(errors.shape[1]):
            error = errors[:, run_id].reshape((-1,))
            aucs.append(metrics.auc(runtime, error) / (denom - start_t))
            minerrors.append(np.min(error))
        result['auc'] = aucs
        result['minerror'] = minerrors
        result['num_entries'] = num_entries


def load_results(params, base_path, scheduler='*', searcher='*', postfix='*'):
    benchmark_name = _path_experiment(params)
    instance_resource = _path_instance_resource(params)
    path_prefix = Path(base_path).joinpath(benchmark_name, instance_resource)
    results = dict()
    glob_expr = '{}/{}/{}/run_*.csv'.format(scheduler, searcher, postfix)
    for p in path_prefix.glob(glob_expr):
        p_parts = p.parts[-4:-1]
        name = '/'.join(p_parts)
        if name not in results:
            results[name] = {'runtime': [], 'error': []}
        df = pd.read_csv(p)
        results[name]['runtime'].append(list(df['runtime']))
        results[name]['error'].append(list(df['best']))
    return results


def load_result(params, base_path, abbrev_keys=True):
    results_path, _ = make_results_path(
        params, base_path, mkdir=False, abbrev_keys=abbrev_keys)
    result = {'runtime': [], 'error': []}
    for p in results_path.glob("run_*.csv"):
        df = pd.read_csv(p)
        result['runtime'].append(list(df['runtime']))
        result['error'].append(list(df['best']))
    return result, results_path


def analyze_results(params, filters=None):
    """
    Utility function to analyze results stored using naming convention given
    by `make_results_path` (e.g., `make_training_history_callback`).

    `filters` is a list of dicts, each with keys 'scheduler', 'searcher',
    'postfix', values are glob patterns. Results to be analyzed is union of
    files matched by each filter.

    :param params:
    :param filters:

    """
    if filters is None:
        filters = [{'scheduler': '*', 'searcher': '*', 'postfix': '*'}]
    results = None
    for filter in filters:
        add_results = load_results(
            params, base_path=params['base_path'],
            scheduler=filter['scheduler'], searcher=filter['searcher'],
            postfix=filter['postfix'])
        if results is None:
            results = add_results
        else:
            for k, v in add_results.items():
                results[k] = results.get(k, v)

    print("Loaded results for {} experiments".format(len(results)))
    compute_experiment_scores(results, max_error=float(params['max_error']))
    results_sorted = sorted([
        (np.mean(result['auc']),
         np.mean(result['minerror']),
         np.std(result['auc']),
         np.std(result['minerror']),
         np.mean(result['num_entries']),
         np.std(result['num_entries']),
         name) for name, result in results.items()])

    print("Name: AUC (+-), minerror (+-), num_entries (+-)\n"
          "-----------------------------------------------")
    for i, (m_auc, m_err, s_auc, s_err, m_num, s_num, name) in enumerate(
            results_sorted):
        print(f"{i+1} {name}:\n"
              f"   {m_auc} (+- {s_auc}), {m_err} (+- {s_err}),"
              f" {m_num} (+- {s_num})")
