from typing import Iterable, List, Optional
import pytest

from autogluon.core.searcher.bayesopt.datatypes.hp_ranges import \
    HyperparameterRanges_Impl, HyperparameterRangeContinuous
from autogluon.core.searcher.bayesopt.datatypes.scaling import LinearScaling
from autogluon.core.searcher.bayesopt.datatypes.common import StateIdAndCandidate
from autogluon.core.searcher.bayesopt.tuning_algorithms.base_classes import \
    ScoringFunction, SurrogateModel
from autogluon.core.searcher.bayesopt.tuning_algorithms.bo_algorithm import \
    _pick_from_locally_optimized, _lazily_locally_optimize, _order_candidates
from autogluon.core.searcher.bayesopt.tuning_algorithms.bo_algorithm_components \
    import NoOptimization
from autogluon.core.searcher.bayesopt.utils.duplicate_detector import \
    DuplicateDetectorIdentical, DuplicateDetectorEpsilon


def test_pick_from_locally_optimized():
    duplicate_detector1 = DuplicateDetectorIdentical()
    duplicate_detector2 = DuplicateDetectorEpsilon(
        hp_ranges=HyperparameterRanges_Impl(
            HyperparameterRangeContinuous('hp1', -10.0, 10.0, scaling=LinearScaling()),
            HyperparameterRangeContinuous('hp2', -10.0, 10.0, scaling=LinearScaling()),
        )
    )
    for duplicate_detector in (duplicate_detector1, duplicate_detector2):

        got = _pick_from_locally_optimized(
            candidates_with_optimization=[
                # original,   optimized
                ((0.1, 1.0), (0.1, 1.0)),
                ((0.1, 1.0), (0.6, 1.0)),  # not a duplicate
                ((0.2, 1.0), (0.1, 1.0)),  # duplicate optimized; Resolved by the original
                ((0.1, 1.0), (0.1, 1.0)),  # complete duplicate
                ((0.3, 1.0), (0.1, 1.0)),  # blacklisted original
                ((0.4, 3.0), (0.3, 1.0)),  # blacklisted all
                ((1.0, 2.0), (1.0, 1.0)),  # final candidate to be selected into a batch
                ((0.0, 2.0), (1.0, 0.0)),  # skipped
                ((0.0, 2.0), (1.0, 0.0)),  # skipped
            ],
            blacklisted_candidates={
                (0.3, 1.0),
                (0.4, 3.0),
                (0.0, 0.0),  # blacklisted candidate, not present in candidates
            },
            num_candidates=4,
            duplicate_detector=duplicate_detector,
        )

        expected = [
            (0.1, 1.0),
            (0.6, 1.0),
            (0.2, 1.0),
            (1.0, 1.0)
        ]

        # order of the candidates should be preserved
        assert len(expected) == len(got)
        assert all(a == b for a, b in zip(got, expected))


def test_lazily_locally_optimize():
    original_candidates = [
        (1.0, 'a', 3, 'b'),
        (2.0, 'c', 2, 'a'),
        (0.0, 'd', 0, 'd')
    ]

    # NoOptimization class is used to check the interfaces only in here
    i = 0
    for candidate in  _lazily_locally_optimize(
            original_candidates, NoOptimization(None, None, None), model=None):
        # no optimization is applied ot the candidates
        assert candidate[0] == original_candidates[i]
        assert candidate[1] == original_candidates[i]
        i += 1

    assert i == len(original_candidates)
    assert len(list(_lazily_locally_optimize(
        [], NoOptimization(None, None, None), model=None))) == 0


@pytest.mark.parametrize('example,expected', [
    (
        {
            (1.0, 'a'),
            (2.0, 'a'),
            (1.0, 'b'),
            (3.0, 'a'),
            (2.0, 'b'),
            (3.0, 'b')
        },
        (1.0, 1.0, 2.0, 2.0, 3.0, 3.0)
    ),
    (
        {
            (1.0, 'a', 'b'),
            (1.0, 'c', 'b'),
            (3.0, 'a', 'd'),
            (0.0, 'e', 'b')
        },
        (0.0, 1.0, 1.0, 3.0)
    ),
    (
        {
            ('ba',),
            ('b',),
            ('a',),
        },
        ('a', 'b', 'ba')
    ),
])
def test_order_candidates(example, expected):
    # Scorer orders by the first dimension
    class MyScorer(ScoringFunction):
        def score(self, candidates: Iterable[StateIdAndCandidate],
                  model: Optional[SurrogateModel] = None) -> List[float]:
            return [candidate[0] for candidate in candidates]

    # very important condition: ordering is deterministic.
    # if not, it breaks fixing random seed in a very sinister way.
    # fixes a bug with random seed, introduced by arbitrary order of set().
    example = list(example)
    A = tuple(_order_candidates(example, MyScorer(), None))
    B = tuple(_order_candidates(example, MyScorer(), None))
    C = tuple(_order_candidates(example, MyScorer(), None))

    assert A == B
    assert A == C
    assert expected == tuple(r[0] for r in A)
