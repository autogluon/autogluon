""" Utility functions for predict_table_column.fit """

import os, multiprocessing
import mxnet as mx
from datetime import datetime

def setup_outputdir(output_directory):
    if output_directory is None:
        timestamp = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
        output_directory = "autogluon-fit-"+timestamp
        os.makedirs(output_directory)
        print("No output_directory specified. Models will be saved in: %s" % output_directory)
    output_directory = os.path.expanduser(output_directory) # replace ~ with absolute path if it exists
    if output_directory[-1] != '/':
        output_directory = output_directory + '/'
    return output_directory

def setup_compute(nthreads_per_trial, ngpus_per_trial):
    if nthreads_per_trial is None:
        nthreads_per_trial = multiprocessing.cpu_count()  # Use all of processing power / trial by default. To use just half: # int(np.floor(multiprocessing.cpu_count()/2))
    if ngpus_per_trial is None:
        if mx.test_utils.list_gpus():
            ngpus_per_trial = 1  # Single GPU / trial
        else:
            ngpus_per_trial = 0
    if ngpus_per_trial > 1:
        ngpus_per_trial = 1
        print("predict_table_column currently does not use more than 1 GPU per training run. ngpus_per_trial has been set = 1")
    return (nthreads_per_trial, ngpus_per_trial)
    
def setup_trial_limits(time_limits, num_trials, hyperparameters={'NN': None}):
    """ Adjust default time limits / num_trials """
    if num_trials is None:
        if time_limits is None:
            time_limits = 10 * 60  # run for 10min by default
        if time_limits <= 20:  # threshold = 20sec, ie. too little time to run >1 trial.
            num_trials = 1
        else:
            num_trials = 1000  # run up to 1000 trials (or as you can within the given time_limits)
    elif time_limits is None:
        time_limits = int(1e6)  # user only specified num_trials, so run all of them regardless of time-limits
    time_limits *= 0.9  # reduce slightly to account for extra time overhead
    time_limits /= float(len(hyperparameters.keys()))  # each model type gets half the available time
    return (time_limits, num_trials)
