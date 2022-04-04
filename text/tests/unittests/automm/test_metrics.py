import pytest
import random
import torch
from autogluon.text.automm.optimization.utils import get_metric
from torchmetrics import MeanMetric
from sklearn.metrics import log_loss


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

    _, _, custom_metric_func = get_metric(metric_name=metric_name)
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
