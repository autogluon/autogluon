# Use STS Benchmark as an example to demonstrate ONNX export and evaluation

import argparse

import onnxruntime as ort
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances

from autogluon.multimodal import MultiModalPredictor


def eval_cosine(predictor, df, onnx_session):
    labels = df["score"].to_numpy()
    valid_input = [
        "hf_text_text_token_ids",
        "hf_text_text_valid_length",
        "hf_text_text_segment_ids",  # Remove for mpnet
    ]

    QEmb = onnx_session.run(None, predictor._learner.get_processed_batch_for_deployment(data=df[["sentence1"]]))[0]
    AEmb = onnx_session.run(None, predictor._learner.get_processed_batch_for_deployment(data=df[["sentence2"]]))[0]

    cosine_scores = 1 - (paired_cosine_distances(QEmb, AEmb))
    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    print(eval_pearson_cosine)
    print(eval_spearman_cosine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name", default="sentence-transformers/msmarco-MiniLM-L-12-v3", type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.model_path:
        args.model_path = args.checkpoint_name.replace("/", "_") + ".onnx"

    # Load Dataset
    val_df = load_dataset("wietsedv/stsbenchmark", split="validation").to_pandas()
    test_df = load_dataset("wietsedv/stsbenchmark", split="test").to_pandas()

    # Init Predictor
    predictor = MultiModalPredictor(
        problem_type="feature_extraction",
        hyperparameters={
            "model.hf_text.checkpoint_name": args.checkpoint_name,
        },
    )

    # Export ONNX model
    onnx_path = predictor.export_onnx(data=val_df, path=args.model_path, verbose=args.verbose)

    # Load ONNX model
    ort_sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])

    # Evaluate ONNX model
    eval_cosine(predictor, test_df, ort_sess)
