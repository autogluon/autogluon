import os
import shutil
import tempfile
import uuid

import numpy.testing as npt
import torch
from torchmetrics import RetrievalHitRate

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import BINARY, MULTICLASS, QUERY, RESPONSE, UNIFORM_SOUP
from autogluon.multimodal.utils import convert_data_for_ranking


def get_home_dir():
    """Get home directory"""
    _home_dir = os.path.join("~", ".automm_unit_tests")
    # expand ~ to actual path
    _home_dir = os.path.expanduser(_home_dir)
    return _home_dir


def get_data_home_dir():
    """Get home directory for storing the datasets"""
    home_dir = get_home_dir()
    return os.path.join(home_dir, "datasets")


def get_repo_url():
    """Return the base URL for Gluon dataset and model repository"""
    repo_url = "https://automl-mm-bench.s3.us-east-1.amazonaws.com/unit-tests-0.4/datasets/"
    if repo_url[-1] != "/":
        repo_url = repo_url + "/"
    return repo_url


def verify_predictor_save_load(predictor, df, verify_embedding=True, cls=MultiModalPredictor):
    root = str(uuid.uuid4())
    os.makedirs(root, exist_ok=True)
    predictor.save(root)
    predictions = predictor.predict(df, as_pandas=False)
    # Test fit_summary()
    predictor.fit_summary()

    loaded_predictor = cls.load(root)
    # Test fit_summary()
    loaded_predictor.fit_summary()

    predictions2 = loaded_predictor.predict(df, as_pandas=False)
    predictions2_df = loaded_predictor.predict(df, as_pandas=True)
    npt.assert_equal(predictions, predictions2)
    npt.assert_equal(predictions2, predictions2_df.to_numpy())
    if predictor.problem_type in [BINARY, MULTICLASS]:
        predictions_prob = predictor.predict_proba(df, as_pandas=False)
        predictions2_prob = loaded_predictor.predict_proba(df, as_pandas=False)
        predictions2_prob_df = loaded_predictor.predict_proba(df, as_pandas=True)
        npt.assert_equal(predictions_prob, predictions2_prob)
        npt.assert_equal(predictions2_prob, predictions2_prob_df.to_numpy())
    if verify_embedding:
        embeddings = predictor.extract_embedding(df)
        assert embeddings.shape[0] == len(df)
    shutil.rmtree(root)


def verify_predictor_realtime_inference(predictor, df, verify_embedding=True):
    for i in range(1, 3):
        df_small = df.head(i)
        predictions_default = predictor.predict(df_small, as_pandas=False, realtime=False)
        predictions_realtime = predictor.predict(df_small, as_pandas=False, realtime=True)
        npt.assert_equal(predictions_default, predictions_realtime)
        if predictor.problem_type in [BINARY, MULTICLASS]:
            predictions_prob_default = predictor.predict_proba(df_small, as_pandas=False, realtime=False)
            predictions_prob_realtime = predictor.predict_proba(df_small, as_pandas=False, realtime=True)
            npt.assert_equal(predictions_prob_default, predictions_prob_realtime)
        if verify_embedding:
            embeddings_default = predictor.extract_embedding(df_small, realtime=False)
            embeddings_realtime = predictor.extract_embedding(df_small, realtime=True)
            npt.assert_equal(embeddings_default, embeddings_realtime)


def verify_no_redundant_model_configs(predictor):
    model_names = list(predictor._learner._config.model.keys())
    model_names.remove("names")
    assert sorted(predictor._learner._config.model.names) == sorted(model_names)


def verify_predict_and_predict_proba(test_data, predictor):
    preds = predictor.predict(test_data)
    proba = predictor.predict_proba(test_data, as_pandas=False)
    assert len(proba) == len(test_data)
    assert (proba.argmax(axis=1) == preds).all()


def verify_predict_as_pandas_and_multiclass(test_data, predictor):
    pandas_pred = predictor.predict(test_data, as_pandas=True)
    pandas_proba = predictor.predict_proba(test_data, as_pandas=True)
    pandas_proba_as_multiclass = predictor.predict_proba(test_data, as_pandas=True, as_multiclass=True)
    pandas_proba_no_multiclass = predictor.predict_proba(test_data, as_pandas=True, as_multiclass=False)

    proba_as_multiclass = predictor.predict_proba(test_data, as_pandas=False, as_multiclass=True)
    proba_as_multiclass = predictor.predict_proba(test_data, as_pandas=False, as_multiclass=True)


def verify_predict_without_label_column(test_data, predictor, label_col="label"):
    test_data_no_label_col = test_data.drop(columns=[label_col], axis=1)
    preds = predictor.predict(test_data_no_label_col)
    assert len(preds) == len(test_data)
    preds2 = predictor.predict(test_data)
    assert len(preds2) == len(test_data)
    assert (preds == preds2).all()
    return preds


def verify_matcher_save_load(matcher, df, verify_embedding=True, cls=MultiModalPredictor):
    with tempfile.TemporaryDirectory() as root:
        matcher.save(root)
        predictions = matcher.predict(df, as_pandas=False)
        loaded_matcher = cls.load(root)
        predictions2 = loaded_matcher.predict(df, as_pandas=False)
        predictions2_df = loaded_matcher.predict(df, as_pandas=True)
        npt.assert_equal(predictions, predictions2)
        npt.assert_equal(predictions2, predictions2_df.to_numpy())
        if matcher.problem_type.endswith((BINARY, MULTICLASS)):
            print("\nverifying predict and predict_proba...\n")
            predictions_prob = matcher.predict_proba(df, as_pandas=False)
            predictions2_prob = loaded_matcher.predict_proba(df, as_pandas=False)
            predictions2_prob_df = loaded_matcher.predict_proba(df, as_pandas=True)
            npt.assert_equal(predictions_prob, predictions2_prob)
            npt.assert_equal(predictions2_prob, predictions2_prob_df.to_numpy())
        if verify_embedding:
            query_embeddings = matcher.extract_embedding(df, signature=QUERY)
            response_embeddings = matcher.extract_embedding(df, signature=RESPONSE)
            assert query_embeddings.shape[0] == len(df)
            assert response_embeddings.shape[0] == len(df)


def verify_matcher_realtime_inference(matcher, df, verify_embedding=True):
    for i in range(1, 3):
        df_small = df.head(i)
        predictions_default = matcher.predict(df_small, as_pandas=False, realtime=False)
        predictions_realtime = matcher.predict(df_small, as_pandas=False, realtime=True)
        npt.assert_equal(predictions_default, predictions_realtime)
        if matcher.problem_type.endswith((BINARY, MULTICLASS)):
            predictions_prob_default = matcher.predict_proba(df_small, as_pandas=False, realtime=False)
            predictions_prob_realtime = matcher.predict_proba(df_small, as_pandas=False, realtime=True)
            npt.assert_almost_equal(predictions_prob_default, predictions_prob_realtime, decimal=5)
        if verify_embedding:
            embeddings_default = matcher.extract_embedding(df_small, signature=QUERY, realtime=False)
            embeddings_realtime = matcher.extract_embedding(df_small, signature=QUERY, realtime=True)
            npt.assert_equal(embeddings_default, embeddings_realtime)
            embeddings_default = matcher.extract_embedding(df_small, signature=RESPONSE, realtime=False)
            embeddings_realtime = matcher.extract_embedding(df_small, signature=RESPONSE, realtime=True)
            npt.assert_equal(embeddings_default, embeddings_realtime)


def evaluate_matcher_ranking(matcher, test_df, query_column, response_column, metric_name, symmetric=False):
    test_df_with_label, test_query_text_data, test_response_image_data, test_label_column = convert_data_for_ranking(
        data=test_df,
        query_column=query_column,
        response_column=response_column,
    )
    socre_1 = matcher.evaluate(
        data=test_df_with_label,
        query_data=test_query_text_data,
        response_data=test_response_image_data,
        metrics=[metric_name],
        label=test_label_column,
        cutoffs=[1, 5, 10],
    )

    if symmetric:
        (
            test_df_with_label,
            test_query_image_data,
            test_response_text_data,
            test_label_column,
        ) = convert_data_for_ranking(
            data=test_df,
            query_column=response_column,
            response_column=query_column,
        )
        socre_2 = matcher.evaluate(
            data=test_df_with_label,
            query_data=test_query_image_data,
            response_data=test_response_text_data,
            metrics=[metric_name],
            label=test_label_column,
            cutoffs=[1, 5, 10],
        )


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
