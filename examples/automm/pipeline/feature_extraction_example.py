# Use STS Benchmark as an example to demonstrate feature extraction pipeline

import argparse
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances
from torch import tensor

from autogluon.multimodal import MultiModalPredictor


def evaluate(predictor, df, onnx_session=None):
    labels = df["score"].to_numpy()

    # TODO: line below only outputs one embedding since dataloader merge text columns automatically
    # mixEmb =  predictor.extract_embedding(valid_df[["sentence1","sentence2"]])["sentence1_sentence2"]
    if not onnx_session:
        QEmb = predictor.extract_embedding(df[["sentence1"]])["sentence1"]
        AEmb = predictor.extract_embedding(df[["sentence2"]])["sentence2"]
    else:
        valid_input = [
            "hf_text_text_token_ids",
            "hf_text_text_valid_length",
            "hf_text_text_segment_ids",
        ]
        QEmb = onnx_session.run(
            None, predictor.get_processed_batch_for_deployment(data=df[["sentence1"]], valid_input=valid_input)
        )[0]
        AEmb = onnx_session.run(
            None, predictor.get_processed_batch_for_deployment(data=df[["sentence2"]], valid_input=valid_input)
        )[0]

    cosine_scores = 1 - paired_cosine_distances(QEmb, AEmb)
    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    print(eval_pearson_cosine)
    print(eval_spearman_cosine)

    return eval_pearson_cosine, eval_spearman_cosine


def main(args):
    ### Dataset Loading
    train_df = load_dataset("wietsedv/stsbenchmark", split="train").to_pandas()
    val_df = load_dataset("wietsedv/stsbenchmark", split="validation").to_pandas()
    test_df = load_dataset("wietsedv/stsbenchmark", split="test").to_pandas()

    start = time.time()
    predictor = MultiModalPredictor(
        pipeline="feature_extraction",
        hyperparameters={
            "model.hf_text.checkpoint_name": args.checkpoint_name,
        },
    )
    ag_load = time.time()
    evaluate(predictor, test_df)
    ag_eval = time.time()

    ort_sess = ort.InferenceSession(args.onnx_path, providers=["CUDAExecutionProvider"])
    onnx_load = time.time()
    evaluate(predictor, test_df, ort_sess)
    onnx_eval = time.time()

    print("Autogluon load time: %.4f" % (ag_load - start))
    print("Autogluon eval time: %.4f" % (ag_eval - ag_load))
    print("ONNX load time: %.4f" % (onnx_load - ag_eval))
    print("ONNX eval time: %.4f" % (onnx_eval - onnx_load))

    exit()
    # TODO: support fit after predict for two tower models:
    predictor.fit(
        train_df,
        val_df,
        hyperparameters={
            "optim.max_epochs": 1,
        },
    )
    evaluate(predictor, test_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name", default="sentence-transformers/msmarco-MiniLM-L-12-v3", type=str)
    parser.add_argument("--onnx_path", default=None, type=str)
    args = parser.parse_args()

    if not args.onnx_path:
        args.onnx_path = "../production/" + args.checkpoint_name.replace("/", "_") + ".onnx"

    main(args)
