from typing import Callable, Tuple, List
import numpy as np

from autogluon.searcher.bayesopt.gpmxnet import SliceException


MAX_STEP_OUT = 200
MAX_STEP_LOOP = 200


class SliceSampler(object):

    def __init__(self, log_density: Callable[[np.ndarray], float], scale: float, random_seed: int):
        self.log_density = log_density
        self.scale = scale  # default in scala core is 1.0
        self.random_state = np.random.RandomState(random_seed)

    def _gen_next_sample(self, x0: np.ndarray) -> np.ndarray:
        random_direction = gen_random_direction(len(x0), self.random_state)

        def sliced_log_density(_movement: float) -> float:
            return self.log_density(x0 + random_direction * _movement)

        # a quantity used to determine the bounds and accept movement along random_direction
        log_pivot = sliced_log_density(0.0) + np.log(self.random_state.rand())

        lower_bound, upper_bound = slice_sampler_step_out(log_pivot, self.scale, sliced_log_density, self.random_state)
        movement = slice_sampler_step_in(lower_bound, upper_bound, log_pivot, sliced_log_density, self.random_state)
        return x0 + random_direction * movement

    def sample(self, init_sample: np.ndarray, num_samples: int, burn: int, thin: int) -> List[np.ndarray]:
        samples = []
        next_sample = init_sample
        for _ in range(num_samples):
            next_sample = self._gen_next_sample(next_sample)
            samples.append(next_sample)
        return samples[burn::thin]


def gen_random_direction(dimension: int, random_state: np.random.RandomState) -> np.ndarray:
    random_direction = random_state.randn(dimension)
    random_direction *= 1.0 / np.linalg.norm(random_direction)
    return random_direction


def slice_sampler_step_out(log_pivot: float, scale: float,
        sliced_log_density: Callable[[float], float], random_state: np.random.RandomState) -> Tuple[float, float]:

    r = random_state.rand()
    lower_bound = -r * scale
    upper_bound = lower_bound + scale

    def bound_step_out(bound, direction):
        """direction -1 for lower bound, +1 for upper bound"""
        for _ in range(MAX_STEP_OUT):
            if sliced_log_density(bound) <= log_pivot:
                return bound
            else:
                bound += direction * scale
        raise SliceException("Reach maximum iteration ({}) while stepping out for bound ({})".format(MAX_STEP_OUT, direction))

    lower_bound = bound_step_out(lower_bound, -1.)
    upper_bound = bound_step_out(upper_bound, 1.)
    return lower_bound, upper_bound


def slice_sampler_step_in(lower_bound: float, upper_bound: float, log_pivot: float,
        sliced_log_density: Callable[[float], float], random_state: np.random.RandomState) -> float:
    """Find the right amount of movement along with a random_direction"""
    for _ in range(MAX_STEP_LOOP):
        movement = (upper_bound - lower_bound) * random_state.rand() + lower_bound
        if movement == 0.0:
            raise SliceException("The interval for slice sampling has reduced to zero in step in")
        if sliced_log_density(movement) > log_pivot:
            return movement
        else:
            lower_bound = movement if movement < 0.0 else lower_bound
            upper_bound = movement if movement > 0.0 else upper_bound
    raise SliceException("Reach maximum iteration ({}) while stepping in".format(MAX_STEP_LOOP))
