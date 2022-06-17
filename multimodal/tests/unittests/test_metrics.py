import pytest
import random
import torch
from torchmetrics import MeanMetric
from sklearn.metrics import log_loss, f1_score
from autogluon.multimodal.optimization.utils import get_metric, get_loss_func
from autogluon.multimodal.constants import MULTICLASS, Y_PRED, Y_TRUE
from autogluon.multimodal.utils import compute_score


@pytest.mark.parametrize(
    "metric_name,class_num",
    [
        ("log_loss", 5),
        ("log_loss", 10),
        ("cross_entropy", 100),
    ]
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

    _, custom_metric_func = get_metric(metric_name=metric_name, problem_type=MULTICLASS)
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
    ]
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


@pytest.mark.parametrize(
    "pos_label",
    [
        0,
        1,
    ]
)
def test_f1(pos_label):
    y_true = [0, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 0, 0, 1, 0]
    score1 = f1_score(y_true, y_pred, pos_label=pos_label)
    metric_data = {
        Y_PRED: y_pred,
        Y_TRUE: y_true,
    }
    score2 = compute_score(
        metric_data=metric_data,
        metric_name="f1",
        pos_label=pos_label,
    )
    assert pytest.approx(score1, 1e-6) == score2

