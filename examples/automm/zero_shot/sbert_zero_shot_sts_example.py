import argparse
from autogluon.multimodal import MultiModalPredictor
from datasets import load_dataset

from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import pearsonr, spearmanr


def main(args):
    ### Dataset Loading
    valid_df = load_dataset("wietsedv/stsbenchmark", split=args.split).to_pandas()
    labels = valid_df["score"].to_numpy()

    predictor = MultiModalPredictor(
        problem_type="sentence_similarity",
        zero_shot=True,
    )

    # TODO: line below only outputs one embedding since dataloader merge text columns automatically
    # mixEmb =  predictor.extract_embedding(valid_df[["sentence1","sentence2"]])["sentence1_sentence2"]

    QEmb = predictor.extract_embedding(valid_df[["sentence1"]])["sentence1"]
    AEmb = predictor.extract_embedding(valid_df[["sentence2"]])["sentence2"]

    cosine_scores = 1 - (paired_cosine_distances(QEmb, AEmb))
    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    print(eval_pearson_cosine)
    print(eval_spearman_cosine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", type=str)
    args = parser.parse_args()

    main(args)
