import logging
import numpy as np
import copy
from typing import NamedTuple, Dict

logger = logging.getLogger(__name__)


class RungEntry(NamedTuple):
    level: int  # Rung level r_j
    prom_quant: float  # Promotion quantile q_j
    data: Dict  # Data of all previous jobs reaching the level


class StoppingRungSystem(object):
    """
    Implements stopping rule resembling the median rule. Once a config is
    stopped, it cannot be promoted later on.
    This is different to what has been published as ASHA (see
    :class:`PromotionRungSystem`).
    """
    def __init__(self, rung_levels, promote_quantiles, max_t):
        # The data entry in _rungs is a dict mapping task_key to
        # reward_value
        assert len(rung_levels) == len(promote_quantiles)
        self._rungs = [
            RungEntry(level=x, prom_quant=y, data=dict())
            for x, y in reversed(list(zip(rung_levels, promote_quantiles)))]

    def on_task_schedule(self):
        return dict()

    def on_task_add(self, task, skip_rungs, **kwargs):
        pass

    @staticmethod
    def _cutoff(recorded, prom_quant):
        if not recorded:
            return None
        return np.percentile(list(recorded.values()), (1 - prom_quant) * 100)

    def on_task_report(self, task, cur_iter, cur_rew, skip_rungs):
        """
        Decision on whether task may continue (task_continues = True), or should be
        stopped (task_continues = False).
        milestone_reached is a flag whether cur_iter coincides with a milestone.
        If True, next_milestone is the next milestone after cur_iter, or None
        if there is none.

        :param task: Only need task.task_id
        :param cur_iter: Current time_attr value of task
        :param cur_rew: Current reward_attr value of task
        :param skip_rungs: This number of lowest rung levels are not
            considered milestones for this task
        :return: dict(task_continues, milestone_reached, next_milestone)
        """
        assert cur_rew is not None, \
            "Reward attribute must be a numerical value, not None"
        task_key = str(task.task_id)
        task_continues = True
        milestone_reached = False
        next_milestone = None
        if skip_rungs > 0:
            milestone_rungs = self._rungs[:(-skip_rungs)]
        else:
            milestone_rungs = self._rungs
        for rung in milestone_rungs:
            milestone = rung.level
            prom_quant = rung.prom_quant
            recorded = rung.data
            if not (cur_iter < milestone or task_key in recorded):
                # Note: It is important for model-based searchers that
                # milestones are reached exactly, not jumped over. In
                # particular, if a future milestone is reported via
                # register_pending, its reward value has to be passed
                # later on via update.
                assert cur_iter == milestone, \
                    "cur_iter = {} > {} = milestone. Make sure to report time attributes covering all milestones".format(
                        cur_iter, milestone)
                milestone_reached = True
                cutoff = self._cutoff(recorded, prom_quant)
                if cutoff is not None and cur_rew < cutoff:
                    task_continues = False
                recorded[task_key] = cur_rew
                break
            next_milestone = milestone
        return {
            'task_continues': task_continues,
            'milestone_reached': milestone_reached,
            'next_milestone': next_milestone}

    def on_task_remove(self, task):
        pass

    def get_first_milestone(self, skip_rungs):
        return self._rungs[-(skip_rungs + 1)].level

    def get_milestones(self, skip_rungs):
        if skip_rungs > 0:
            milestone_rungs = self._rungs[:(-skip_rungs)]
        else:
            milestone_rungs = self._rungs
        return [x.level for x in milestone_rungs]

    def snapshot_rungs(self):
        return copy.deepcopy(self._rungs)

    def __repr__(self):
        iters = " | ".join([
            "Iter {:.3f}: {}".format(
                r.level, self._cutoff(r.data, r.prom_quant))
            for r in self._rungs])
        return "Rung system: " + iters
