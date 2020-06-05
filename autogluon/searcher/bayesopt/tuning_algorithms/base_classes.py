from abc import ABC, abstractmethod
from typing import List, Iterator, Iterable, Tuple, Type, Optional
import mxnet as mx
import numpy as np
from mxnet.ndarray import NDArray

from autogluon.searcher.bayesopt.datatypes.common import Candidate
from autogluon.searcher.bayesopt.datatypes.tuning_job_state import \
    TuningJobState


class NextCandidatesAlgorithm:
    def next_candidates(self) -> List[Candidate]:
        # Not using ABC otherwise it will be difficult to create a subclass that also is a
        # NamedTuple(as both define their own metaclass)
        raise NotImplemented("Abstract method")


class CandidateGenerator(ABC):
    """
    Class to generate candidates from which to start the local minimization, typically random candidate
    or some form of more uniformly spaced variation, such as latin hypercube or sobol sequence
    """
    @abstractmethod
    def generate_candidates(self) -> Iterator[Candidate]:
        pass

    def generate_candidates_en_bulk(self, num_cands: int) -> List[Candidate]:
        raise NotImplementedError()


class SurrogateModel(ABC):
    def __init__(self, state: TuningJobState, active_metric: str,
                 random_seed: int):
        self.state = state
        self.random_seed = random_seed
        self.active_metric = active_metric

    @abstractmethod
    def predict_nd(self, x_nd: NDArray) -> List[Tuple[NDArray, NDArray]]:
        """
        Given a (n, d) matrix x_nd of test input points, return predictive means
        and predictive stddevs, as (n,) vectors.

        Note: Different to state passed at construction, the input points in
        x_nd must be encoded already. See also 'predict_candidates'.

        If the model supports fantasizing (see FantasizingSurrogateModel), and
        the state passed at construction contains pending evaluations with
        fantasized target values, then pred_mean will be a matrix of shape
        (n, num_fantasy_samples), one column per fantasy sample, while pred_std
        remains a vector (the same for each sample).

        When using GP with marginal likelihood estimation, the returned list
        will be of length 1. When using GP with MCMC, the returned list will have
        one entry per MCMC sample.

        :param x_nd: Test input points
        :return: A list of (pred_mean, pred_std)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Wrapper around predict_nd, where inputs and returns are np.ndarray.

        Note: Different to state passed at construction, the input points in
        X must be encoded already. See also 'predict_candidates'.
        """
        pass

    def predict_candidates(self, candidates: Iterable[Candidate]) -> \
            List[Tuple[np.ndarray, np.ndarray]]:
        """
        Convenience function to get a list of means and standard deviations
        of candidates.
        """
        return self.predict(self.state.hp_ranges.to_ndarray_matrix(candidates))

    @abstractmethod
    def current_best(self) -> np.ndarray:
        """
        Returns the so-called incumbent, to be used in acquisition functions
        such as expected improvement. This is the minimum of predictive means
        at all current candidate locations (both state.candidate_evaluations
        and state.pending_evaluations).
        Normally, a scalar is returned, but if the model supports fantasizing
        and the state contains pending evaluations, there is one incumbent
        per fantasy sample, so a vector is returned.

        NOTE: When using MCMC, we should really maintain one incumbent per MCMC
        sample (for the same reason as we do that for fantasies). This is
        currently not done.

        :return: Incumbent
        """
        pass

    @abstractmethod
    def context_for_nd(self) -> mx.Context:
        """
        :return: Context for mx.nd input/output arguments
        """
        pass

    @abstractmethod
    def dtype_for_nd(self):
        """
        :return: Datatype for mx.nd input/output arguments
        """
        pass

    def convert_np_to_nd(self, x: np.ndarray) -> NDArray:
        return mx.nd.array(
            x, ctx=self.context_for_nd(), dtype=self.dtype_for_nd())


class ScoringFunction(ABC):
    """
    Class to score candidates, typically combine an acquisition function with
    potentially Thompson sampling

    NOTE: it will be minimized, i.e. lower is better
    """
    @abstractmethod
    def score(self, candidates: Iterable[Candidate],
              model: Optional[SurrogateModel] = None) -> List[float]:
        """
        Requires multiple candidates, is this can be much quicker: we can use matrix operations

        lower is better
        """
        pass


class AcquisitionFunction(ScoringFunction):
    @abstractmethod
    def __init__(self, model: SurrogateModel):
        self.model = model

    @abstractmethod
    def compute_acq(self, x: np.ndarray,
                    model: Optional[SurrogateModel] = None) -> np.ndarray:
        pass

    @abstractmethod
    def compute_acq_with_gradients(
            self, x: np.ndarray,
            model: Optional[SurrogateModel] = None) -> \
            Tuple[np.ndarray, np.ndarray]:
        pass

    def score(self, candidates: Iterable[Candidate],
              model: Optional[SurrogateModel] = None) -> List[float]:
        if model is None:
            model = self.model
        x = model.state.hp_ranges.to_ndarray_matrix(candidates)
        return list(self.compute_acq(x, model=model))


class LocalOptimizer(ABC):
    """
    Class that tries to find a local candidate with a better score, typically using a local
    optimization method such as lbfgs. It would normally encapsulate an acquisition function and model

    """
    def __init__(self, state: TuningJobState, model: SurrogateModel,
                 acquisition_function_class: Type[AcquisitionFunction]):
        self.state = state
        self.model = model
        self.acquisition_function_class = acquisition_function_class

    @abstractmethod
    def optimize(self, candidate: Candidate,
                 model: Optional[SurrogateModel] = None) -> Candidate:
        """
        Run local optimization, starting from candidate.
        If model is given, it overrides self.model.

        :param candidate: Starting point
        :param model: See above
        :return: Candidate found by local optimization
        """
        pass


class PendingCandidateStateTransformer(ABC):
    """
    This concept is needed if HPO can deal with pending candidates, which
    remain unlabeled during further decision-making (e.g., batch decisions
    or asynchronous HPO). In this case, information for the pending candidate
    has to be added to the state.

    """
    @abstractmethod
    def append_candidate(self, candidate: Candidate):
        """
        Determines PendingEvaluation information for candidate and append to
        state.pending_evaluations.

        :param candidate: Novel pending candidate
        """
        pass

    @property
    @abstractmethod
    def state(self) -> TuningJobState:
        """
        :return: Current TuningJobState
        """
        pass

    @abstractmethod
    def model(self, **kwargs) -> SurrogateModel:
        """
        In general, the model is computed here on demand, based on the current
        state, unless the state has not changed since the last 'model' call.

        :return: Surrogate model for current state
        """
        pass
