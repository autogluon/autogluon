import argparse
import json
import os

import pandas as pd
from dataset import (
    AdultTabularDataset,
    AloiTabularDataset,
    CaliforniaHousingTabularDataset,
    CovtypeTabularDataset,
    EpsilonTabularDataset,
    HelenaTabularDataset,
    HiggsSmallTabularDataset,
    JannisTabularDataset,
    MicrosoftTabularDataset,
    YahooTabularDataset,
    YearTabularDataset,
)
from ray import tune

from autogluon.multimodal import MultiModalPredictor

TABULAR_DATASETS = {
    "ad": AdultTabularDataset,
    "al": AloiTabularDataset,
    "ca": CaliforniaHousingTabularDataset,
    "co": CovtypeTabularDataset,
    "ep": EpsilonTabularDataset,
    "he": HelenaTabularDataset,
    "hi": HiggsSmallTabularDataset,
    "ja": JannisTabularDataset,
    "mi": MicrosoftTabularDataset,
    "ya": YahooTabularDataset,
    "ye": YearTabularDataset,
}

automm_hyperparameters = {
    "data.categorical.convert_to_text": False,
    "model.names": ["ft_transformer"],
    "model.ft_transformer.embedding_arch": ["linear"],
    "env.batch_size": 128,
    "env.per_gpu_batch_size": 128,
    "env.inference_batch_size_ratio": 1,
    "env.num_workers": 12,
    "env.num_workers_inference": 12,
    "env.num_gpus": 1,
    "optim.max_epochs": 2000,  # Specify a large value to train until convergence
    "optim.weight_decay": 1.0e-5,
    "optim.lr_choice": None,
    "optim.lr_schedule": "polynomial_decay",
    "optim.warmup_steps": 0.0,
    "optim.patience": 20,
    "optim.top_k": 3,
}

hyperparameter_tune_kwargs = {
    "searcher": "random",
    "scheduler": "FIFO",
    "num_trials": 50,
}


def main(args):
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    assert args.dataset_name in TABULAR_DATASETS.keys(), "Unsupported dataset name."

    ### Dataset loading
    train_data = TABULAR_DATASETS[args.dataset_name]("train", args.dataset_dir)

    val_data = TABULAR_DATASETS[args.dataset_name]("val", args.dataset_dir)

    test_data = TABULAR_DATASETS[args.dataset_name]("test", args.dataset_dir)

    automm_hyperparameters["optim.lr"] = args.lr
    automm_hyperparameters["optim.end_lr"] = args.end_lr

    if args.embedding_arch is not None:
        automm_hyperparameters["model.ft_transformer.embedding_arch"] = args.embedding_arch

    tabular_hyperparameters = {
        "GBM": [
            {},
            {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
        ],
        "CAT": {},
        "XGB": {},
        "AG_AUTOMM": automm_hyperparameters,
    }

    if args.mode == "single":
        ### model initialization
        predictor = MultiModalPredictor(
            label=train_data.label_column,
            problem_type=train_data.problem_type,
            eval_metric=train_data.metric,
            path=args.exp_dir,
            verbosity=4,
        )

        ### model training
        predictor.fit(
            train_data=train_data.data,
            tuning_data=val_data.data,
            seed=args.seed,
            hyperparameters=automm_hyperparameters,
        )

        ### model inference
        scores = predictor.evaluate(data=test_data.data, metrics=[test_data.metric])
        with open(os.path.join(args.exp_dir, "scores.json"), "w") as f:
            json.dump(scores, f)
        print(scores)
    elif args.mode == "single_hpo":
        automm_hyperparameters["model.ft_transformer.ffn_dropout"] = tune.uniform(0.0, 0.5)
        automm_hyperparameters["model.ft_transformer.attention_dropout"] = tune.uniform(0.0, 0.5)
        automm_hyperparameters["model.ft_transformer.residual_dropout"] = tune.uniform(0.0, 0.2)
        automm_hyperparameters["model.ft_transformer.ffn_hidden_size"] = tune.randint(150, 300)
        automm_hyperparameters["optim.lr"] = tune.uniform(0.00001, 0.001)
        automm_hyperparameters["optim.end_lr"] = 1e-5

        ### model initialization
        predictor = MultiModalPredictor(
            label=train_data.label_column,
            problem_type=train_data.problem_type,
            eval_metric=train_data.metric,
            path=args.exp_dir,
            verbosity=4,
        )

        ### model training
        predictor.fit(
            train_data=train_data.data,
            tuning_data=val_data.data,
            seed=args.seed,
            hyperparameters=automm_hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        )

        ### model inference
        scores = predictor.evaluate(data=test_data.data, metrics=[test_data.metric])
        with open(os.path.join(args.exp_dir, "scores.json"), "w") as f:
            json.dump(scores, f)
        print(scores)
    elif args.mode == "weighted" or args.mode == "single_bag5" or args.mode == "stack5":
        if args.mode == "single_bag5":
            tabular_hyperparameters = {
                "AG_AUTOMM": automm_hyperparameters,
            }
            num_bag_folds, num_stack_levels = 5, 0
        elif args.mode == "weighted":
            num_bag_folds, num_stack_levels = None, None
        elif args.mode == "stack5":
            num_bag_folds, num_stack_levels = 5, 1
        else:
            raise NotImplementedError
        from autogluon.tabular import TabularPredictor

        predictor = TabularPredictor(eval_metric=train_data.metric, label=train_data.label_column, path=args.exp_dir)
        predictor.fit(
            train_data=train_data.data,
            tuning_data=val_data.data if num_bag_folds is None else None,
            hyperparameters=tabular_hyperparameters,
            num_bag_folds=num_bag_folds,
            num_stack_levels=num_stack_levels,
        )
        leaderboard = predictor.leaderboard()
        leaderboard.to_csv(os.path.join(args.exp_dir, "leaderboard.csv"))
    else:
        raise NotImplementedError
    scores = predictor.evaluate(data=test_data.data)
    with open(os.path.join(args.exp_dir, "scores.json"), "w") as f:
        json.dump(scores, f)
    print(scores)

    predictions = predictor.predict(data=test_data.data)
    predictions.to_csv(os.path.join(args.exp_dir, "predictions.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=None, type=str, help="Specify the GPU to use.")
    parser.add_argument("--dataset_name", default="ad", type=str, help="Specify the dataset to run the experinments.")
    parser.add_argument("--dataset_dir", default="./dataset", type=str, help="Path to the dataset.")
    parser.add_argument("--exp_dir", default=None, type=str, help="Path to the outputs.")
    parser.add_argument("--lr", default=1e-04, type=float, help="Initial learning rate.")
    parser.add_argument("--end_lr", default=1e-04, type=float, help="End learning rate.")
    parser.add_argument(
        "--mode",
        choices=["single", "single_hpo", "weighted", "single_bag5", "stack5"],
        default="single",
        help="Method to run with.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--embedding_arch",
        type=str,
        nargs="+",
        default=None,
        help="Embedding architecture for numerical features in FT_Transformer.",
    )
    args = parser.parse_args()

    if args.exp_dir is None:
        args.exp_dir = f"./results/{args.dataset_name}"

    main(args)
