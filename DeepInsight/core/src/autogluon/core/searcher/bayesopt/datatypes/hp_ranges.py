from abc import ABC, abstractmethod
from math import floor
from typing import Tuple, List, Iterable, Dict
import numpy as np

from .common import Hyperparameter, Candidate
from .scaling import Scaling

# Epsilon margin to account for numerical errors
EPS = 1e-8


class HyperparameterRange(ABC):
    """
    One can argue that this class does a bit too much: both definition of
    ranges and things like normalization, for now we do not plan to make the
    latter configurable, so it should be okay having everything in one place to
    keeps things fairly simple
    """
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def to_ndarray(self, hp: Hyperparameter) -> np.ndarray:
        pass

    @abstractmethod
    def from_ndarray(self, cand_ndarray: np.ndarray) -> Hyperparameter:
        pass

    def ndarray_size(self) -> int:
        return 1

    @abstractmethod
    def from_zero_one(self, normalized_value: float) -> Hyperparameter:
        """
        Used to generate a random hp, takes as input a random value between 0.0
        and 1.0, which should be generated uniformly at random. Returns a valid
        active hyperparameter.

        Normally is the same as from_ndarray, but in the warm start case it is
        not. Will be limited to active ranges.
        """
        # by default this is the same as from_ndarray
        pass

    def random_hp(self, random_state) -> Hyperparameter:
        return self.from_zero_one(random_state.uniform(0.0, 1.0))


def scale_from_zero_one(
        value, lower_bound: float, upper_bound: float, scaling: Scaling):
    assert 0.0 <= value <= 1.0, value
    if lower_bound == upper_bound:
        return lower_bound
    else:
        lower = scaling.to_internal(lower_bound)
        upper = scaling.to_internal(upper_bound)
        range = upper - lower
        assert range > 0, (lower, upper)
        internal_value = value * range + lower
        hp = scaling.from_internal(internal_value)
        # set value in case it is off due to numerical rounding
        if hp < lower_bound:
            hp = lower_bound
        if hp > upper_bound:
            hp = upper_bound
        return hp


class HyperparameterRangeContinuous(HyperparameterRange):
    def __init__(
            self, name: str, lower_bound: float, upper_bound: float,
            scaling: Scaling, active_lower_bound: float = None,
            active_upper_bound: float = None):
        """
        Real valued hyperparameter.

        :param name: unique name of the hyperparameter.
        :param lower_bound: inclusive lower bound on all the values that
            parameter can take.
        :param upper_bound: inclusive upper bound on all the values that
            parameter can take.
        :param scaling: determines how the values of the parameter are enumerated internally.
            The parameter value is expressed as parameter = scaling(internal), where internal
            is internal representation of parameter, whcih is a real value, normally in range
            [0, 1]. To optimize the parameter, the internal is varied, and the parameter to be
            tested is calculated from such internal representation. This allows to speed up
            on many machine learning problems, where the parameter being optimized adjusts
            modelling capacity of the model. In such situation, often holds that
            capacity = log(parameter). See e.g. VC dimension for finite set-family:
            https://en.wikipedia.org/wiki/Vapnik–Chervonenkis_dimension#Bounds
        :param active_lower_bound: changes lower bound of the range of the parameter; Is used
            within warmstart functionality.
        :param active_upper_bound: changes upper bound on the values of parameter. Used for
            warmstart.
        """
        super().__init__(name)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.active_lower_bound = lower_bound if active_lower_bound is None else active_lower_bound
        self.active_upper_bound = upper_bound if active_upper_bound is None else active_upper_bound

        assert self.lower_bound <= self.active_upper_bound <= self.upper_bound
        assert self.lower_bound <= self.active_lower_bound <= self.upper_bound

        self.scaling = scaling

    def to_ndarray(self, hp: float) -> np.ndarray:
        assert isinstance(hp, float), (hp, self)
        assert self.lower_bound <= hp <= self.upper_bound, (hp, self)
        # convert everything to internal scaling, and then normalize between zero and one
        lower = self.scaling.to_internal(self.lower_bound)
        upper = self.scaling.to_internal(self.upper_bound)
        if upper == lower:
            result = 0.0  # if the bounds are fixed for a dimension
        else:
            result = (self.scaling.to_internal(hp) - lower) / (upper - lower)
        assert 0.0 <= result <= 1.0, (result, self)
        return np.array([result])

    def from_ndarray(self, ndarray: np.ndarray) -> float:
        scalar = ndarray.item()
        return scale_from_zero_one(scalar, self.lower_bound,
                                   self.upper_bound, self.scaling)

    def __repr__(self) -> str:
        return "{}({}, {}, {}, {}, {}, {})".format(
            self.__class__.__name__, repr(self.name),
            repr(self.active_lower_bound), repr(self.active_upper_bound),
            repr(self.scaling), repr(self.lower_bound), repr(self.upper_bound)
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HyperparameterRangeContinuous):
            return self.name == other.name \
                   and self.lower_bound == other.lower_bound \
                   and self.upper_bound == other.upper_bound \
                   and self.scaling == other.scaling
        return False

    def from_zero_one(self, normalized_value: float) -> float:
        return scale_from_zero_one(normalized_value, self.active_lower_bound,
                                   self.active_upper_bound, self.scaling)


class HyperparameterRangeInteger(HyperparameterRange):
    def __init__(self, name: str, lower_bound: int, upper_bound: int, scaling: Scaling,
                 active_lower_bound: int = None, active_upper_bound: int = None):
        """
        Both bounds are INCLUDED in the valid values. Under the hood generates a continuous
        range from lower_bound - 0.5 to upper_bound + 0.5.
        See docs for continuous hyperparameter for more information.
        """
        super().__init__(name)
        # reduce the range by epsilon at both ends, to avoid corner cases where numerical rounding
        # would cause a value that would end up out of range by one
        active_lower_bound = lower_bound if active_lower_bound is None \
            else active_lower_bound
        active_upper_bound = upper_bound if active_upper_bound is None \
            else active_upper_bound

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.active_lower_bound = active_lower_bound
        self.active_upper_bound = active_upper_bound
        self.scaling = scaling

        self._continuous_range = HyperparameterRangeContinuous(
            name, lower_bound - 0.5 + EPS, upper_bound + 0.5 - EPS, scaling,
            active_lower_bound - 0.5 + EPS, active_upper_bound + 0.5 - EPS,
        )

    def to_ndarray(self, hp: int) -> np.ndarray:
        assert isinstance(hp, int), (hp, type(hp), self)
        return self._continuous_range.to_ndarray(float(hp))

    def from_ndarray(self, ndarray: np.ndarray) -> int:
        continuous = self._continuous_range.from_ndarray(ndarray)
        result = int(round(continuous))
        # just to be sure
        assert result >= self.lower_bound, (result, self)
        assert result <= self.upper_bound, (result, self)
        return result

    def __repr__(self) -> str:
        return "{}({}, {}, {}, {}, {}, {})".format(
            self.__class__.__name__, repr(self.name),
            repr(self.active_lower_bound), repr(self.active_upper_bound),
            repr(self.scaling), repr(self.lower_bound), repr(self.upper_bound)
        )

    def __eq__(self, other):
        if isinstance(other, HyperparameterRangeInteger):
            return self.name == other.name \
                   and self.lower_bound == other.lower_bound \
                   and self.upper_bound == other.upper_bound \
                   and self.active_lower_bound == other.active_lower_bound \
                   and self.active_upper_bound == other.active_upper_bound \
                   and self.scaling == other.scaling
        return False

    def from_zero_one(self, normalized_value: float) -> Hyperparameter:
        scaling_result = scale_from_zero_one(
            normalized_value, self._continuous_range.active_lower_bound,
            self._continuous_range.active_upper_bound, self.scaling)
        result = int(round(scaling_result))
        return result


class HyperparameterRangeCategorical(HyperparameterRange):
    def __init__(self, name: str, choices: Tuple[str, ...],
                 active_choices: Tuple[str, ...] = None):
        """
        Can take on discrete set of values.
        :param name: name of dimension.
        :param choices: possible values of the hyperparameter.
        :param active_choices: a subset of choices, restricts choices to some subset of values.
        """
        super().__init__(name)
        # sort the choices, so that we are sure the order is the same when we do one hot encoding
        choices = sorted(choices)
        self.choices = choices
        self.active_choices = choices if active_choices is None else active_choices
        self.active_choices = sorted(self.active_choices)

        assert set(self.choices).issuperset(self.active_choices)

    def to_ndarray(self, hp: str) -> np.ndarray:
        assert hp in self.choices, "{} not in {}".format(hp, self)
        idx = self.choices.index(hp)
        result = np.zeros(shape=(len(self.choices),))
        result[idx] = 1.0
        return result

    def from_ndarray(self, cand_ndarray: np.ndarray) -> str:
        assert len(cand_ndarray) == len(self.choices), (cand_ndarray, self)
        return self.choices[int(np.argmax(cand_ndarray))]

    def ndarray_size(self) -> int:
        return len(self.choices)

    def from_zero_one(self, normalized_value: float) -> str:
        if normalized_value == 1.0:
            return self.active_choices[-1]
        else:
            return self.active_choices[int(floor(normalized_value * len(self.active_choices)))]

    def __repr__(self) -> str:
        return "{}({}, {})".format(
            self.__class__.__name__, repr(self.name), repr(self.choices)
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, HyperparameterRangeCategorical):
            return self.name == other.name \
                   and self.choices == other.choices
        return False


class HyperparameterRanges(ABC):
    @abstractmethod
    def to_ndarray(self, cand_tuple: Candidate,
                   categ_onehot: bool = True) -> np.ndarray:
        """
        Categorical values are one-hot encoded if categ_onehot is true,
        otherwise they are mapped to 0, 1, ...

        :param cand_tuple: HP tuple to encode
        :param categ_onehot: See above. Def: True
        :return: Encoded HP vector
        """
        pass

    def to_ndarray_matrix(self, candidates: Iterable[Candidate],
                          categ_onehot: bool = True) -> np.ndarray:
        return np.vstack(
            [self.to_ndarray(cand, categ_onehot) for cand in candidates])

    @abstractmethod
    def ndarray_size(self) -> int:
        pass

    @abstractmethod
    def from_ndarray(self, cand_ndarray: np.ndarray) -> Candidate:
        """
        Converts a candidate from internal ndarray representation (fed to the
        GP) into an external Candidate. This typically involves rounding.

        For numerical HPs it assumes values scaled between 0.0 and 1.0, for
        categorical HPs it assumes one scalar per category, which will convert
        to the category with the highest value (think of it as a soft one hot
        encoding).
        """
        pass

    @abstractmethod
    def random_candidate(self, random_state) -> Candidate:
        pass

    @abstractmethod
    def random_candidates(self, random_state, num_configs: int) -> \
            List[Candidate]:
        pass

    @abstractmethod
    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        """
        Returns (lower, upper) bounds for each dimension in ndarray vector
        representation.

        :return: [(lower0, upper0), (lower1, upper1), ...]
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    # Needed because some code iterates over HyperparameterRange entries
    @property
    def hp_ranges(self) -> Tuple[HyperparameterRange]:
        raise NotImplementedError(
            "Only some HyperparameterRanges subclasses implement hp_ranges")


class HyperparameterRanges_Impl(HyperparameterRanges):
    """
    Alternative to HyperparameterRanges_CS, without depending on ConfigSpace.

    """
    def __init__(self, *args: HyperparameterRange):
        """
        Usage:

        hp_ranges = HyperparameterRanges_Impl(hp1_range, hp2_range, hp3_range)
        """
        self._hp_ranges = tuple(args)

        names = [hp_range.name for hp_range in args]
        if len(set(names)) != len(names):
            raise ValueError("duplicate names in {}".format(names))

    def to_ndarray(self, cand_tuple: Candidate,
                   categ_onehot: bool = True) -> np.ndarray:
        if not categ_onehot:
            raise NotImplementedError("categ_onehot = False not implemented")
        pieces = [hp_range.to_ndarray(hp) for hp_range, hp in zip(self._hp_ranges, cand_tuple)]
        return np.hstack(pieces)

    def ndarray_size(self) -> int:
        return sum(d.ndarray_size() for d in self._hp_ranges)

    def from_ndarray(self, cand_ndarray: np.ndarray) -> Candidate:
        """
        Converts a candidate from internal ndarray representation (fed to the GP) into an
        external Candidate

        For numerical HPs it assumes values scaled between 0.0 and 1.0, for categorical HPs it
        assumes one scalar per category, which will convert to the category with the highest value
        (think of it as a soft one hot encoding).

        all such values should be concatenated, see to_ndarray for the opposite conversion
        """

        # check size
        expected_size = self.ndarray_size()
        assert cand_ndarray.shape == (expected_size,), (cand_ndarray.shape, expected_size)

        hps = []
        start = 0
        for hp_range in self._hp_ranges:
            end = start + hp_range.ndarray_size()
            hps.append(hp_range.from_ndarray(cand_ndarray[start:end]))
            start = end

        return tuple(hps)

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        # Context values are fixed by setting min = max in bounds of lbfgs minimizer:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.Bounds.html#scipy.optimize.Bounds
        bounds = []
        for hp_range in self._hp_ranges:
            if isinstance(hp_range, HyperparameterRangeCategorical):
                for category in hp_range.choices:
                    if category in hp_range.active_choices:
                        if len(hp_range.active_choices) == 1:
                            bounds.append((1.0, 1.0))  # constrain choice to the only one which is active
                        else:
                            bounds.append((0.0, 1.0))
                    else:
                        bounds.append((0.0, 0.0))
            else:
                # conversion below includes log / reverse log / etc scaling
                # due to warm start the below could be different from global
                # bounds
                low = hp_range.to_ndarray(hp_range.from_zero_one(0.0)).item()
                high = hp_range.to_ndarray(hp_range.from_zero_one(1.0)).item()
                bounds.append((low, high))
        return bounds

    def refine_ndarray_bounds(
            self, bounds: List[Tuple[float, float]], candidate: Candidate,
            margin: float) -> List[Tuple[float, float]]:
        new_bounds = []
        bound_it = iter(bounds)
        for i, hp_range in enumerate(self._hp_ranges):
            if isinstance(hp_range, HyperparameterRangeCategorical):
                for _ in range(len(hp_range.choices)):
                    new_bounds.append(next(bound_it))
            else:
                low, high = next(bound_it)
                x = hp_range.to_ndarray(candidate[i]).item()
                low = max(low, x - margin)
                high = min(high, x + margin)
                new_bounds.append((low, high))
        assert len(bounds) == len(new_bounds)  # Sanity check
        return new_bounds

    def to_kwargs(self, cand_tuple: Candidate) -> Dict[str, Hyperparameter]:
        return {hp_range.name: hp for hp_range, hp in zip(self._hp_ranges, cand_tuple)}

    def random_candidate(self, random_state) -> Candidate:
        return tuple(hp_range.random_hp(random_state) for hp_range in self._hp_ranges)

    def random_candidates(self, random_state, num_configs: int) -> \
            List[Candidate]:
        return [self.random_candidate(random_state)
                for _ in range(num_configs)]

    def __repr__(self) -> str:
        return "{}{}".format(
            self.__class__.__name__, repr(self._hp_ranges)
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HyperparameterRanges_Impl):
            return self._hp_ranges == other.hp_ranges
        return False

    def __len__(self) -> int:
        return len(self._hp_ranges)

    @property
    def hp_ranges(self) -> Tuple[HyperparameterRange]:
        return self._hp_ranges
