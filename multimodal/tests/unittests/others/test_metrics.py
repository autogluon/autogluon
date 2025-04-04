import os
import random
import shutil
import tempfile

import pytest
import torch
from sklearn.metrics import f1_score, log_loss
from torchmetrics import MeanMetric, RetrievalHitRate

import autogluon.core.metrics as ag_metrics
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import MULTICLASS, Y_PRED, Y_TRUE
from autogluon.multimodal.optimization.utils import compute_hit_rate, get_loss_func, get_metric
from autogluon.multimodal.utils import compute_score, infer_metrics

from ..utils.unittest_datasets import HatefulMeMesDataset, PetFinderDataset
from ..utils.utils import get_home_dir


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


# TODO (1): torchmetrics will give slightly different result under multi GPU runs
# TODO (2): "F1" is not supported for multiclass, will fallback to accuracy
@pytest.mark.single_gpu
@pytest.mark.parametrize(
    "eval_metric",
    ["f1_macro", "f1_micro", "f1_weighted"],
)
def test_f1_metrics_for_multiclass(eval_metric):
    dataset = PetFinderDataset()
    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type="multiclass",
        eval_metric=eval_metric,
    )
    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": ["ft_transformer"],
        "env.num_gpus": 1,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "optimization.top_k_average_method": "best",
        "optimization.loss_function": "auto",
        "data.categorical.convert_to_text": False,  # ensure the categorical model is used.
        "data.numerical.convert_to_text": False,  # ensure the numerical model is used.
    }
    save_path = os.path.join(get_home_dir(), "outputs", eval_metric)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    predictor.fit(
        train_data=dataset.train_df,
        tuning_data=dataset.test_df,
        hyperparameters=hyperparameters,
        time_limit=20,
        save_path=save_path,
    )
    val_score = predictor._learner._best_score
    eval_score = predictor.evaluate(dataset.test_df)[eval_metric]
    assert abs(val_score - eval_score) < 2e-2


@pytest.mark.single_gpu
@pytest.mark.parametrize(
    "problem_type, eval_metric_name, validation_metric_name, is_matching, target_eval_metric_name, target_validation_metric_name",
    [
        ("binary", "acc", None, False, "acc", "acc"),
        ("multiclass", "f1_macro", None, False, "f1_macro", "f1_macro"),
        ("regression", "f1", None, False, "rmse", "rmse"),
        ("object_detection", "map", "map", False, "map", "map"),
        ("binary", None, None, False, "roc_auc", "roc_auc"),
        ("multiclass", None, None, False, "accuracy", "accuracy"),
        ("regression", None, None, False, "rmse", "rmse"),
        ("object_detection", None, None, False, "map", "map"),
        ("semantic_segmentation", None, None, False, "iou", "iou"),
        ("ner", None, None, False, "overall_f1", "ner_token_f1"),
        ("few_shot_classification", None, None, False, "accuracy", "accuracy"),
        ("binary", None, None, True, "roc_auc", "roc_auc"),
        ("multiclass", None, None, True, "spearmanr", "spearmanr"),
        ("regression", None, None, True, "spearmanr", "spearmanr"),
        ("regression", "pearsonr", None, True, "pearsonr", "pearsonr"),
        (None, None, None, True, "ndcg", "recall"),
        ("feature_extraction", None, None, False, None, None),
        ("feature_extraction", "f1", None, False, None, None),
    ],
)
def test_infer_metrics(
    problem_type,
    eval_metric_name,
    validation_metric_name,
    is_matching,
    target_eval_metric_name,
    target_validation_metric_name,
):
    validation_metric_name, eval_metric_name = infer_metrics(
        problem_type, eval_metric_name, validation_metric_name, is_matching
    )
    assert eval_metric_name == target_eval_metric_name
    assert validation_metric_name == target_validation_metric_name


# Once eval metric is customized, shall not use the fallback eval
@pytest.mark.single_gpu
@pytest.mark.parametrize(
    "problem_type, eval_metric, is_matching, target_eval_metric_name, target_validation_metric_name",
    [
        ("binary", ag_metrics.make_scorer("dummy", ag_metrics.get_metric("acc")), False, "dummy", "roc_auc"),
        ("regression", ag_metrics.make_scorer("dummy", ag_metrics.get_metric("acc")), True, "dummy", "spearmanr"),
    ],
)
def test_infer_metrics_custom(
    problem_type,
    eval_metric,
    is_matching,
    target_eval_metric_name,
    target_validation_metric_name,
):
    validation_metric_name, eval_metric_name = infer_metrics(problem_type, eval_metric, None, is_matching)
    assert eval_metric_name == target_eval_metric_name
    assert validation_metric_name == target_validation_metric_name


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


def test_custom_metric():
    dataset = HatefulMeMesDataset()
    metric_name = dataset.metric
    custom_metric_name = "customized"
    metric_scorer = ag_metrics.get_metric(metric_name)
    metric_scorer.name = custom_metric_name
    predictor_by_name = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    predictor_by_scorer = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_scorer,
    )
    with tempfile.TemporaryDirectory() as save_path:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        predictor_by_name.fit(
            train_data=dataset.train_df,
            time_limit=0,
            save_path=save_path,
        )
    with tempfile.TemporaryDirectory() as save_path:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        predictor_by_scorer.fit(
            train_data=dataset.train_df,
            time_limit=0,
            save_path=save_path,
        )
    scores_by_name = predictor_by_name.evaluate(
        data=dataset.test_df,
        metrics=None,
    )
    scores_by_scorer_eval = predictor_by_name.evaluate(
        data=dataset.test_df,
        metrics=[metric_scorer],
    )
    scores_by_scorer_init = predictor_by_scorer.evaluate(
        data=dataset.test_df,
        metrics=None,
    )
    assert scores_by_scorer_eval[custom_metric_name] == scores_by_scorer_init[custom_metric_name]
    assert scores_by_name[metric_name] == scores_by_scorer_eval[custom_metric_name]
