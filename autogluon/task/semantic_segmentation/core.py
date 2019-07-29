
from gluoncv.model_zoo.segbase import get_segmentation_model

from ...basic import autogluon_register
from ...basic.space import *
from .pipeline import *
from ...scheduler import * 
from ...searcher import *
from ..base import Results, BaseTask

class SemanticSegmentation(BaseTask):
    def fit(self,
            trainset,
            valset,
            searcher=RandomSampling,
            scheduler=FIFO_Scheduler,
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
        searcher = searcher(config_space)
        task_scheduler = scheduler(
                train_fn=train_semantic_segmentation,
                args=train_semantic_segmentation.args,
                resurce=resource,
                searcher=searcher,
            )
        task_scheduler.run()
        task_scheduler.join_tasks()
