import numpy as np
import ConfigSpace as CS
from typing import List, Tuple

from ..datatypes.common import Candidate
from ..datatypes.hp_ranges import HyperparameterRanges


class HyperparameterRanges_CS(HyperparameterRanges):
    def __init__(self, config_space: CS.ConfigurationSpace,
                 name_last_pos: str = None,
                 value_for_last_pos = None):
        """
        If name_last_pos is given, the hyperparameter of that name is assigned
        the final position in the vector returned by to_ndarray. This can be
        used to single out the (time) resource for a GP model, where that
        component has to come last.

        If in this case (name_last_pos given), value_for_last_pos is also given,
        some methods are modified:
        - random_candidate samples a config as normal, but then overwrites the
          name_last_pos component by value_for_last_pos
        - get_ndarray_bounds works as normal, but returns bound (a, a) for
          name_last_pos component, where a is the internal value corresponding
          to value_for_last_pos
        The use case is HPO with a resource attribute. This attribute should be
        fixed when optimizing the acquisition function, but can take different
        values in the evaluation data (coming from all previous searches).

        :param config_space: ConfigurationSpace
        :param name_last_pos: See above. Default: None
        :param value_for_last_pos: See above. Default: None
        """
        self.config_space = config_space
        self.name_last_pos = name_last_pos
        self.value_for_last_pos = value_for_last_pos
        # Supports conversion to ndarray
        numer_src = []
        numer_trg = []
        categ_src = []
        categ_trg = []
        categ_card = []
        trg_pos = 0
        append_at_end = None
        for src_pos, hp in enumerate(config_space.get_hyperparameters()):
            if isinstance(hp, CS.CategoricalHyperparameter):
                card = hp.num_choices
                if hp.name == name_last_pos:
                    assert append_at_end is None
                    append_at_end = (src_pos, card, True)
                else:
                    categ_src.append(src_pos)
                    categ_trg.append(trg_pos)
                    categ_card.append(card)
                    trg_pos += card
            elif isinstance(hp, CS.UniformIntegerHyperparameter) or \
                    isinstance(hp, CS.UniformFloatHyperparameter):
                if hp.name == name_last_pos:
                    assert append_at_end is None
                    append_at_end = (src_pos, 1, False)
                else:
                    numer_src.append(src_pos)
                    numer_trg.append(trg_pos)
                    trg_pos += 1
            else:
                raise NotImplementedError(
                    "We only support hyperparameters of type "
                    "CategoricalHyperparameter, UniformIntegerHyperparameter, "
                    "UniformFloatHyperparameter")
        if append_at_end is not None:
            if append_at_end[2]:
                categ_src.append(append_at_end[0])
                categ_trg.append(trg_pos)
                categ_card.append(append_at_end[1])
            else:
                numer_src.append(append_at_end[0])
                numer_trg.append(trg_pos)
            trg_pos += append_at_end[1]
        self.numer_src = np.array(numer_src, dtype=np.int64)
        self.numer_trg = np.array(numer_trg, dtype=np.int64)
        self.categ_src = np.array(categ_src, dtype=np.int64)
        self.categ_trg = np.array(categ_trg, dtype=np.int64)
        self.categ_card = np.array(categ_card, dtype=np.int64)
        self._ndarray_size = trg_pos

    def to_ndarray(self, cand_tuple: Candidate) -> np.ndarray:
        assert isinstance(cand_tuple, CS.Configuration)
        trgvec = np.zeros((self._ndarray_size,))
        srcvec = cand_tuple.get_array()
        # https://wesmckinney.com/blog/numpy-indexing-peculiarities/
        # take, put much faster than []
        trgvec.put(
            self.numer_trg, srcvec.take(self.numer_src, mode='clip'),
            mode='clip')
        relpos = srcvec.take(self.categ_src, mode='clip').astype(np.int64)
        trgvec.put(self.categ_trg + relpos, [1.], mode='clip')
        return trgvec

    def ndarray_size(self) -> int:
        return self._ndarray_size

    def from_ndarray(self, cand_ndarray: np.ndarray) -> Candidate:
        assert cand_ndarray.size == self._ndarray_size, \
            "Internal vector [{}] must have size {}".format(
                cand_ndarray, self._ndarray_size)
        cand_ndarray = cand_ndarray.reshape((-1,))
        assert cand_ndarray.min() >= 0. and cand_ndarray.max() <= 1., \
            "Internal vector [{}] must have entries in [0, 1]".format(
                cand_ndarray)
        # Deal with categoricals by using argmax
        srcvec = np.zeros(self.__len__(), dtype=cand_ndarray.dtype)
        srcvec.put(
            self.numer_src, cand_ndarray.take(self.numer_trg, mode='clip'),
            mode='clip')
        for srcpos, trgpos, card in zip(
                self.categ_src, self.categ_trg, self.categ_card):
            maxpos = cand_ndarray[trgpos:(trgpos + card)].argmax()
            srcvec[srcpos] = maxpos
        # Rest is dealt with by CS.Configuration
        return CS.Configuration(self.config_space, vector=srcvec)

    def is_attribute_fixed(self):
        return (self.name_last_pos is not None) and \
               (self.value_for_last_pos is not None)

    def _fix_attribute_value(self, name):
        return self.is_attribute_fixed() and name == self.name_last_pos

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        bounds = []
        final_bound = None
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, CS.CategoricalHyperparameter):
                if not self._fix_attribute_value(hp.name):
                    bound = [(0., 1.)] * len(hp.choices)
                else:
                    bound = [(0., 0.)] * len(hp.choices)
                    bound[int(self.value_for_last_pos)] = (1., 1.)
            else:
                if not self._fix_attribute_value(hp.name):
                    bound = [(0., 1.)]
                else:
                    val_int = float(hp._inverse_transform(
                        np.array([self.value_for_last_pos])).item())
                    bound = [(val_int, val_int)]
            if hp.name == self.name_last_pos:
                final_bound = bound
            else:
                bounds.extend(bound)
        if final_bound is not None:
            bounds.extend(final_bound)
        return bounds

    # NOTE: Assumes that config argument not used afterwards...
    def _transform_config(self, config: CS.Configuration) -> CS.Configuration:
        values = config.get_dictionary()  # No copy is done here
        values[self.name_last_pos] = self.value_for_last_pos
        return CS.Configuration(self.config_space, values=values)

    def random_candidate(self, random_state) -> Candidate:
        self.config_space.random = random_state  # Not great...
        rnd_config = self.config_space.sample_configuration()
        if self.is_attribute_fixed():
            rnd_config = self._transform_config(rnd_config)
        return rnd_config

    def random_candidates(
            self, random_state, num_configs: int) -> List[Candidate]:
        self.config_space.random = random_state  # Not great...
        rnd_configs = self.config_space.sample_configuration(num_configs)
        if self.is_attribute_fixed():
            rnd_configs = [self._transform_config(x) for x in rnd_configs]
        return rnd_configs

    def __repr__(self) -> str:
        return "{}{}".format(
            self.__class__.__name__, repr(self.config_space))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HyperparameterRanges_CS):
            return self.config_space == other.config_space
        return False

    def __len__(self) -> int:
        return len(self.config_space.get_hyperparameters())

    # This is just for convenience, roughly shadows what is done in
    # HyperparameterRanges_Impl
    @property
    def hp_ranges(self) -> List[CS.hyperparameters.Hyperparameter]:
        return self.config_space.get_hyperparameters()

    def filter_for_last_pos_value(
            self, candidates: List[Candidate]) -> List[Candidate]:
        """
        If is_attribute_fixed, the candidates list is filtered by removing
        entries whose name_last_pos attribute value is different from
        value_for_last_pos. Otherwise, candidates is returned unchanged.

        """
        if self.is_attribute_fixed():
            def filter_pred(x: CS.Configuration) -> bool:
                x_dct = x.get_dictionary()
                return (x_dct[self.name_last_pos] == self.value_for_last_pos)

            candidates = list(filter(filter_pred, candidates))
        return candidates
