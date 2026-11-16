import math

import pytest

from autogluon.multimodal.optim.metrics.ranking_metrics import RankingMetrics, compute_ranking_score


@pytest.mark.parametrize("scores", [[0.9, 0.8, 0.7], [9, 8, 7], [-1, -2, -3]])
def test_ranking_metrics_depend_on_order_not_score_scale(scores):
    pred = {"query": dict(zip(["a", "b", "c"], scores))}
    target = {"query": {"a": 1, "c": 1}}

    result = RankingMetrics(pred, target).compute(k=3)

    assert result == pytest.approx(
        {
            "precision@3": 2 / 3,
            "recall@3": 1,
            "mrr@3": 1,
            "map@3": (1 + 2 / 3) / 2,
            "ndcg@3": (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)),
        }
    )


@pytest.mark.parametrize("target", [{}, {"a": 0, "b": -1}])
def test_ranking_metrics_without_relevant_documents(target):
    result = RankingMetrics({"query": {"a": 0.9}}, {"query": target}).compute(k=3)

    assert result == {f"{metric}@3": 0 for metric in ["precision", "recall", "mrr", "map", "ndcg"]}


def test_ranking_metrics_average_queries_without_relevant_documents():
    pred = {"positive": {"a": 0.5}, "negative": {"a": 0.5}}
    target = {"positive": {"a": 1}, "negative": {"a": 0}}

    result = compute_ranking_score(pred, target, metrics=["map", "recall", "ndcg", "precision"], cutoffs=[1, 2])

    assert result == {
        "map@1": 0.5,
        "recall@1": 0.5,
        "ndcg@1": 0.5,
        "precision@1": 0.5,
        "map@2": 0.5,
        "recall@2": 0.5,
        "ndcg@2": 0.5,
        "precision@2": 0.25,
    }


def test_ranking_metrics_ascending_scores():
    pred = {"query": {"irrelevant": 3, "relevant": -2}}
    target = {"query": {"relevant": 1}}

    result = RankingMetrics(pred, target, is_higher_better=False).compute(k=1)

    assert all(value == 1 for value in result.values())


@pytest.mark.parametrize("pred", [{}, {"irrelevant": 0.9}])
def test_ranking_metrics_without_hits(pred):
    result = RankingMetrics({"query": pred}, {"query": {"relevant": 1}}).compute(k=2)

    assert all(value == 0 for value in result.values())


@pytest.mark.parametrize("k", [0, -1])
def test_ranking_metrics_require_positive_cutoff(k):
    with pytest.raises(ValueError, match="k must be positive"):
        RankingMetrics({"query": {"a": 1}}, {"query": {"a": 1}}).compute(k=k)
