"""Fine-tune Mitra through TabularPredictor.

Example:
    python example_mitra_v2.py train.csv --label target --time-limit 3600
    python example_mitra_v2.py train.csv --label target --model-version 1
    python example_mitra_v2.py train.csv --label target --model-version 1 --fine-tune-recipe 1
"""

import argparse

from autogluon.tabular import TabularDataset, TabularPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv")
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-version", choices=["1", "2"], default="2")
    parser.add_argument(
        "--fine-tune-recipe", choices=["1", "2"], help="Defaults to recipe 2 for either checkpoint version"
    )
    parser.add_argument("--checkpoint", help="Optional Hugging Face repo ID or local saved checkpoint directory")
    parser.add_argument("--num-bag-folds", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=3600)
    args = parser.parse_args()
    hyperparameters = {
        "model_version": args.model_version,
        "fine_tune_recipe": args.fine_tune_recipe,
        "ag.max_time_limit": args.time_limit,
        "ag.ens.ag.max_time_limit": args.time_limit,
    }
    if args.checkpoint:
        hyperparameters["hf_model"] = args.checkpoint
    predictor = TabularPredictor(label=args.label).fit(
        TabularDataset(args.train_csv),
        hyperparameters={"MITRA": hyperparameters},
        num_bag_folds=args.num_bag_folds,
        time_limit=args.time_limit,
    )
    print(predictor.leaderboard())


if __name__ == "__main__":
    main()
