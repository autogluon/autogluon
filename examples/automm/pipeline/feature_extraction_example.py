# Use STS Benchmark as an example to demonstrate feature extraction pipeline

import argparse
from autogluon.multimodal import MultiModalPredictor
from datasets import load_dataset

from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import pearsonr, spearmanr

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import tensor


def eval(predictor, df, onnx_session=None):
    labels = df["score"].to_numpy()

    # TODO: line below only outputs one embedding since dataloader merge text columns automatically
    # mixEmb =  predictor.extract_embedding(valid_df[["sentence1","sentence2"]])["sentence1_sentence2"]
    if not onnx_session:
        QEmb = predictor.extract_embedding(df[["sentence1"]])["sentence1"]
        AEmb = predictor.extract_embedding(df[["sentence2"]])["sentence2"]
    else:
        QEmb = onnx_session.run(None, predictor.get_processed_batch(data=df[["sentence1"]]))[0]
        AEmb = onnx_session.run(None, predictor.get_processed_batch(data=df[["sentence2"]]))[0]

    cosine_scores = 1 - (paired_cosine_distances(QEmb, AEmb))
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

    predictor = MultiModalPredictor(
        pipeline="feature_extraction",
        hyperparameters={
            "model.hf_text.checkpoint_name": args.checkpoint_name,
        },
    )
    eval(predictor, test_df)

    ort_sess = ort.InferenceSession(args.onnx_path, providers=["CUDAExecutionProvider"])
    eval(predictor, test_df, ort_sess)

    # TODO: support fit after predict for two tower models
    predictor.fit(
        train_df,
        val_df,
        hyperparameters={
            "optimization.max_epochs": 1,
        },
    )
    eval(predictor, test_df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name", default="sentence-transformers/msmarco-MiniLM-L-12-v3", type=str)
    parser.add_argument("--onnx_path", default="/media/code/autogluon/examples/automm/production/sentence-transformers_msmarco-MiniLM-L-12-v3.onnx", type=str)
    args = parser.parse_args()
    main(args)
