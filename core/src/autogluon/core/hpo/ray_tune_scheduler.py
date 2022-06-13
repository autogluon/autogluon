import time

from ray.tune.schedulers import FIFOScheduler


class AvgEarlyStopFIFOScheduler(FIFOScheduler):
    
    def __init__(self, time_limit=None, **kwargs):
        super().__init__(**kwargs)
        self.time_limit = time_limit
        assert self.time_limit is None or self.time_limit > 0
        self.time_start = time.time()
        self.prev_times = []

    def on_trial_complete(self, trial_runner, trial, result):
        time_total = result.get('time_total_s', -1)
        if time_total != -1:
            self.prev_times.append(time_total)
        super().on_trial_complete(trial_runner, trial, result)

    def choose_trial_to_run(self, trial_runner):
        if self.time_limit is not None:
            time_elapsed = time.time() - self.time_start
            time_left = self.time_limit - time_elapsed
            if self.avg_time > time_left:
                return None
        return super().choose_trial_to_run(trial_runner)
            
    @property
    def avg_time(self):
        if len(self.prev_times) > 0:
            return sum(self.prev_times) / len(self.prev_times)
        return float('-inf')
