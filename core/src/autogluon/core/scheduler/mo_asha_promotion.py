import logging
import numpy as np

from .hyperband_promotion import PromotionRungSystem

logger = logging.getLogger(__name__)

# exploration strategies
EPS_NET = "eps_net"
NSGA_II = "nsga_ii"


class MOPromotionRungSystem(PromotionRungSystem):
    r"""This is an adaption of the standard PromotionRungSystem class to the
    multi-objective domain. Its main functions are eps-net and nsga-ii filters.

    Arguments and handling follow the parent class. One addtional parameter is
    to specify the exploration strategy.

    Args:
        strategy : string
            Indicates the explorations strategy. Supported strategies are
            "eps_net" and "nsga_ii".
    """

    def __init__(self, rung_levels, promote_quantiles, max_t, strategy):
        assert strategy in [EPS_NET, NSGA_II], "Invalid selection strategy"
        self.strategy = strategy
        super().__init__(rung_levels, promote_quantiles, max_t)

    def _find_promotable_config(self, recorded, prom_quant, config_key=None):
        """
        Scans the top prom_quant fraction of recorded (sorted w.r.t. reward
        value) for config not yet promoted. If config_key is given, the key
        must also be equal to config_key.

        :param recorded: Dict to scan
        :param prom_quant: Quantile for promotion
        :param config_key: See above
        :return: Key of config if found, otherwise None
        """
        num_recorded = len(recorded)
        ret_key = None
        if num_recorded >= int(round(1.0 / prom_quant)):
            # Search for not yet promoted config in the top
            # prom_quant fraction
            def filter_pred(k, v):
                return (not v[1]) and (config_key is None or k == config_key)

            num_top = int(num_recorded * prom_quant)

            # generate matrix from individual reward vectors
            evaluations = np.array([v[0] for _, v in recorded.items()])

            if self.strategy == EPS_NET:
                ranked_top = get_eps_net_ranking(evaluations, num_top)
            else:  # NSGA_II
                ranked_top = get_nsga_ii_ranking(evaluations, num_top)

            items = list(recorded.items())
            top_list = [items[i] for i in ranked_top]

            try:
                ret_key = next(k for k, v in top_list if filter_pred(k, v))
            except StopIteration:
                ret_key = None
        return ret_key

    @staticmethod
    def _num_promotable_config(recorded, prom_quant):
        raise NotImplementedError()


# -----------------------------------------------------------------------------
# Filtration techniques
# -----------------------------------------------------------------------------

def fast_nondominated_sort(values: np.array, num_samples: int):
    # We assume a 2d np array of dim (n_candidates, n_objectives)
    # This functions assumes a minimization problem. Implementation is based
    # on the NSGA-II paper
    assert values.shape[0] > 0, "Need to provide at least 1 point."
    assert values.shape[0] >= num_samples, "Need at least enough points to \
                                            meet num_samples."
    domination_counts = np.zeros(values.shape[0])
    dominated_solutions = [[] for _ in range(values.shape[0])]
    fronts = [[]]
    ranks = np.zeros(values.shape[0])

    for i, v1 in enumerate(values):
        for j, v2 in enumerate(values):
            if np.alltrue(v1 < v2):  # v1 dominates v2
                dominated_solutions[i].append(j)
            elif np.alltrue(v2 < v1):  # v2 dominates v1
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            ranks[i] = 1
            fronts[0].append(i)

    i = 0
    n_selected = len(fronts[0])
    while n_selected < num_samples:
        assert len(fronts[i]) > 0, "Cannot select from empty front"
        tmp = []
        for j in fronts[i]:
            for k in dominated_solutions[j]:
                domination_counts[k] -= 1
                if domination_counts[k] == 0:
                    ranks[k] = i + 1
                    tmp.append(k)
        i += 1
        n_selected += len(tmp)
        fronts.append(tmp)
    assert n_selected >= num_samples, "Could not assign enough samples"
    return ranks, fronts


def compute_eps_net(points: np.array, num_samples: int = None):
    """Sparsify a numpy matrix of points returning `num_samples` points that
    are as spread out as possible. Iteratively select the point that is the
    furthest from priorly selected points.
    :param points: numpy array of shape num_points x num_objectives
    :param num_samples: number of points to return
    :return: indices
    """
    assert points.shape[0] > 0, "Need to provide at least 1 point."

    def dist(points, x):
        return np.min([np.linalg.norm(p - x) for p in points])
    n = points.shape[0]
    eps_net = [0]
    indices_remaining = set(range(1, n))
    if num_samples is None:
        num_samples = n
    while len(eps_net) < num_samples and len(indices_remaining) > 0:
        # compute argmin dist(pts[i \not in eps_net], x)
        dist_max = -1
        best_i = 0
        for i in indices_remaining:
            cur_dist = dist(points[eps_net], points[i])
            if cur_dist > dist_max:
                best_i = i
                dist_max = cur_dist
        eps_net.append(best_i)
        indices_remaining.remove(best_i)
    return eps_net


def get_eps_net_ranking(points: np.array, num_top: int):
    """Produces sorted list containing the best indices
    :param points: Numpy array containing all previous evaluations
    :return: List of num_top indices
    """
    _, fronts = fast_nondominated_sort(points, num_top)
    ranked_ids = []

    i = 0
    n_selected = 0
    while n_selected < num_top:
        front = fronts[i]
        local_order = compute_eps_net(points[front],
                                      num_samples=(num_top - n_selected))
        ranked_ids += [front[j] for j in local_order]
        i += 1
        n_selected += len(local_order)
    assert len(ranked_ids) == num_top, "Did not assign correct number of \
                                        points to eps-net"
    return ranked_ids


def crowding_distance_assignment(front_points: np.array):
    assert front_points.shape[0] > 0, "Error no empty fronts are allowed"
    distances = np.zeros(front_points.shape[0])
    n_objectives = front_points.shape[1]

    for m in range(n_objectives):  # Iterate through objectives
        vs = [(front_points[i][m], i) for i in range(front_points.shape[0])]
        vs.sort()
        # last and first element have inf distance
        distances[vs[0][1]] = np.inf
        distances[vs[-1][1]] = np.inf
        ms = [front_points[i][m] for i in range(front_points.shape[0])]
        scale = max(ms) - min(ms)
        if scale == 0:
            scale = 1
        for j in range(1, front_points.shape[0] - 1):
            distances[vs[j][1]] += (vs[j + 1][0] - vs[j - 1][0]) / scale

    # determine local order
    dist_id = [(-distances[i], i) for i in range(front_points.shape[0])]
    dist_id.sort()
    local_order = [d[1] for d in dist_id]
    return local_order


def get_nsga_ii_ranking(points: np.array, num_top: int):
    """Produces sorted list containing the best indices
    :param points: Numpy array containing all previous evaluations
    :return: List of num_top indices
    """
    _, fronts = fast_nondominated_sort(points, num_top)
    ranked_ids = []

    i = 0
    n_selected = 0
    while n_selected < num_top:
        front = fronts[i]
        local_order = crowding_distance_assignment(points[front])
        ranked_ids += [front[j] for j in local_order]
        i += 1
        n_selected += len(local_order)
    return ranked_ids[:num_top]
