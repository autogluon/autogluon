from abc import ABC, abstractmethod
from typing import List, Iterator, Iterable, Tuple, Type, Optional, Set, Dict
import numpy as np

from ..datatypes.common import Candidate
from ..datatypes.tuning_job_state import TuningJobState


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

    @staticmethod
    def keys_predict() -> Set[str]:
        """
        Keys of signals returned by 'predict'.
        Note: In order to work with 'AcquisitionFunction' implementations,
        the following signals are required:

        - 'mean': Predictive mean
        - 'std': Predictive standard deviation

        :return:
        """
        return {'mean', 'std'}

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Given (n, d) matrix of encoded input points, returns signals which are
        statistics of the predictive distribution at these inputs. By default,
        signals are:

        - 'mean': Predictive means. If the model supports fantasizing with a
            number nf of fantasies, this has shape (n, nf), otherwise (n,)
        - 'std': Predictive stddevs, shape (n,)

        If the hyperparameters of the surrogate model are being optimized (e.g.,
        by empirical Bayes), the returned list has length 1. If its
        hyperparameters are averaged over by MCMC, the returned list has one
        entry per MCMC sample.
        """
        pass

    def predict_candidates(self, candidates: Iterable[Candidate]) -> \
            List[Dict[str, np.ndarray]]:
        """
        Convenience variant of 'predict', where the input is a list of n
        candidates, which are encoded to input points here.
        """
        return self.predict(self.state.hp_ranges.to_ndarray_matrix(candidates))

    @abstractmethod
    def current_best(self) -> List[np.ndarray]:
        """
        Returns the so-called incumbent, to be used in acquisition functions
        such as expected improvement. This is the minimum of predictive means
        (signal with key 'mean') at all current candidate locations (both
        state.candidate_evaluations and state.pending_evaluations).
        Normally, a scalar is returned, but if the model supports fantasizing
        and the state contains pending evaluations, there is one incumbent
        per fantasy sample, so a vector is returned.

        If the hyperparameters of the surrogate model are being optimized (e.g.,
        by empirical Bayes), the returned list has length 1. If its
        hyperparameters are averaged over by MCMC, the returned list has one
        entry per MCMC sample.

        :return: Incumbent
        """
        pass

    @abstractmethod
    def backward_gradient(
            self, input: np.ndarray,
            head_gradients: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        """
        Computes the gradient nabla_x f of an acquisition function f(x),
        where x is a single input point. This is using reverse mode
        differentiation, the head gradients are passed by the acquisition
        function.

        If p = p(x) denotes the output of 'predict' for a single input point,
        'head_gradients' contains the head gradients nabla_p f. Its shape is
        that of p (where n=1).

        Lists have >1 entry if MCMC is used, otherwise they are all size 1.

        :param input: Single input point x, shape (d,)
        :param head_gradients: See above
        :return: Gradient nabla_x f (several if MCMC is used)
        """
        pass


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
    def compute_acq(self, inputs: np.ndarray,
                    model: Optional[SurrogateModel] = None) -> np.ndarray:
        """
        Note: If inputs has shape (d,), it is taken to be (1, d)

        :param inputs: Encoded input points, shape (n, d)
        :param model: If given, overrides self.model
        :return: Acquisition function values, shape (n,)
        """
        pass

    @abstractmethod
    def compute_acq_with_gradient(
            self, input: np.ndarray,
            model: Optional[SurrogateModel] = None) -> \
            Tuple[float, np.ndarray]:
        """
        For a single input point x, compute acquisition function value f(x)
        and gradient nabla_x f.

        :param input: Single input point x, shape (d,)
        :param model: If given, overrides self.model
        :return: f(x), nabla_x f
        """
        pass

    def score(self, candidates: Iterable[Candidate],
              model: Optional[SurrogateModel] = None) -> List[float]:
        if model is None:
            model = self.model
        inputs = model.state.hp_ranges.to_ndarray_matrix(candidates)
        return list(self.compute_acq(inputs, model=model))


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
