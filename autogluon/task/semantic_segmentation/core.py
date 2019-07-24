
from gluoncv.model_zoo.segbase import get_segmentation_model

from ...basic import autogluon_register
from ...basic.space import *
from .pipeline import *
from ...scheduler import * 
from ...searcher import *

def fit(dataset, searcher=, scheduler=FIFO_Scheduler,
        resume=False, savedir='checkpoint/exp1.ag', 
        stop_criterion={
                'time_limits': 1 * 60 * 60,
                'max_trial_count': 2
            },
        resource={
                'num_gpus': 0,
                'num_cpus': 4,
            },
        config_space=None,
        **kwars):
    config_space = config_space if config_space else semantic_segmentation_pipelne.cs
    searcher = RandomSampling(config_space)
    scheduler(train_fn=semantic_segmentation_pipelne,
              args=parse_args(), 
              resurce=resource,
              )

