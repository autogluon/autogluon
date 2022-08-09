# Use STS Benchmark as an example to demonstrate feature extraction pipeline

import argparse
from autogluon.multimodal import MultiModalPredictor
from datasets import load_dataset

from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import pearsonr, spearmanr


def eval(predictor, df):
    labels = df["score"].to_numpy()

    # TODO: line below only outputs one embedding since dataloader merge text columns automatically
    # mixEmb =  predictor.extract_embedding(valid_df[["sentence1","sentence2"]])["sentence1_sentence2"]
    QEmb = predictor.extract_embedding(df[["sentence1"]])["sentence1"]
    AEmb = predictor.extract_embedding(df[["sentence2"]])["sentence2"]

    cosine_scores = 1 - (paired_cosine_distances(QEmb, AEmb))
    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    print(eval_pearson_cosine)
    print(eval_spearman_cosine)

    return eval_pearson_cosine, eval_spearman_cosine


def main():
    ### Dataset Loading
    train_df = load_dataset("wietsedv/stsbenchmark", split="train").to_pandas()
    val_df = load_dataset("wietsedv/stsbenchmark", split="validation").to_pandas()
    test_df = load_dataset("wietsedv/stsbenchmark", split="test").to_pandas()

    predictor = MultiModalPredictor(pipeline="feature_extraction")

    eval(predictor, test_df)

    # TODO
    predictor.fit(
        train_df,
        val_df,
        hyperparameters={
            "optimization.max_epochs": 1,
        },
    )
    eval(predictor, test_df)


if __name__ == "__main__":
    main()
