import random

import pytest
import torch
from sklearn.metrics import f1_score, log_loss
from torchmetrics import MeanMetric, RetrievalHitRate

from autogluon.multimodal.constants import MULTICLASS, Y_PRED, Y_TRUE
from autogluon.multimodal.optimization.utils import compute_hit_rate, get_loss_func, get_metric
from autogluon.multimodal.utils import compute_score


@pytest.mark.parametrize(
    "metric_name,class_num",
    [
        ("log_loss", 5),
        ("log_loss", 10),
        ("cross_entropy", 100),
    ],
)
def test_cross_entropy(metric_name, class_num):
    preds = []
    targets = []
    random.seed(123)
    torch.manual_seed(123)

    for i in range(100):
        bs = random.randint(1, 16)
        preds.append(torch.randn(bs, class_num))
        targets.append(torch.randint(0, class_num, (bs,)))

    _, custom_metric_func = get_metric(metric_name=metric_name)
    mean_metric = MeanMetric()

    for per_pred, per_target in zip(preds, targets):
        mean_metric.update(custom_metric_func(per_pred, per_target))

    score1 = mean_metric.compute()
    preds = torch.cat(preds).softmax(dim=1)
    targets = torch.cat(targets)
    score2 = log_loss(
        y_true=targets,
        y_pred=preds,
    )
    assert pytest.approx(score1, 1e-6) == score2


@pytest.mark.parametrize(
    "problem_type,loss_func_name",
    [
        ("regression", "bcewithlogitsloss"),
    ],
)
def test_bce_with_logits_loss(problem_type, loss_func_name):
    preds = []
    targets = []
    random.seed(123)
    torch.manual_seed(123)

    for i in range(100):
        bs = random.randint(1, 16)
        preds.append(torch.randn(bs, 1))
        targets.append(torch.rand(bs, 1))
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    loss_func = get_loss_func(
        problem_type=problem_type,
        mixup_active=False,
        loss_func_name=loss_func_name,
    )

    score1 = loss_func(input=preds, target=targets)
    preds = preds.sigmoid()
    bceloss = torch.nn.BCELoss()
    score2 = bceloss(input=preds, target=targets)
    assert pytest.approx(score1, 1e-6) == score2


def ref_symmetric_hit_rate(features_a, features_b, logit_scale, top_ks=[1, 5, 10]):
    assert len(features_a) == len(features_b)
    hit_rate = 0
    logits_per_a = (logit_scale * features_a @ features_b.t()).detach().cpu()
    logits_per_b = logits_per_a.t().detach().cpu()
    num_elements = len(features_a)
    for logits in [logits_per_a, logits_per_b]:
        preds = logits.reshape(-1)
        indexes = torch.broadcast_to(torch.arange(num_elements).reshape(-1, 1), (num_elements, num_elements)).reshape(
            -1
        )
        target = torch.eye(num_elements, dtype=bool).reshape(-1)
        for k in top_ks:
            hr_k = RetrievalHitRate(top_k=k)
            hit_rate += hr_k(preds, target, indexes=indexes)
    return hit_rate / (2 * len(top_ks))


def test_symmetric_hit_rate():
    generator = torch.Generator()
    generator.manual_seed(0)
    for repeat in range(3):
        for top_ks in [[1, 5, 10], [20], [3, 7, 9]]:
            features_a = torch.randn(50, 2, generator=generator)
            features_b = torch.randn(50, 2, generator=generator)
            hit_rate_impl = compute_hit_rate(features_a, features_b, logit_scale=1.0, top_ks=top_ks)
            hit_rate_ref = ref_symmetric_hit_rate(features_a, features_b, logit_scale=1.0, top_ks=top_ks)
            assert pytest.approx(hit_rate_impl.item()) == hit_rate_ref.item()
