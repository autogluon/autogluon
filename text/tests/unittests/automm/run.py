import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from autogluon.text.automm import AutoMMPredictor
from autogluon.text.automm.constants import (
    MODEL,
    DATA,
    OPTIMIZATION,
    ENVIRONMENT,
    BINARY,
    MULTICLASS,
)
from datasets import (
    PetFinderDataset,
    HatefulMeMesDataset,
    AEDataset,
    SanFranciscoAirbnbDataset,
)
from utils import get_home_dir

ALL_DATASETS = {
    "petfinder": PetFinderDataset,
    "hateful_memes": HatefulMeMesDataset,
    "ae": AEDataset,
    "san_francisco_airbnb": SanFranciscoAirbnbDataset,
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='petfinder', type=str)
    # parser.add_argument('--model', default='fusion_mlp', type=str)
    # parser.add_argument('--data-dir', default='.auto_mm_bench/datasets', type=str)
    parser.add_argument('--exp-dir', default='exp', type=str)
    parser.add_argument('--num-gpus', default=1, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--model-config', default="fusion_mlp_image_text_tabular", type=str)
    parser.add_argument('--data-config', default="default", type=str)
    parser.add_argument('--optim-config', default="adamw", type=str)
    parser.add_argument('--env-config', default="default", type=str)
    parser.add_argument('--ckpt-path', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('overrides', metavar='N', type=str, nargs='*',
                        help='Additional flags to overwrite the configuration')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    # dataset_name = "petfinder"
    dataset = ALL_DATASETS[args.dataset]()
    # print(dataset.feature_columns)
    # exit()
    feature_columns = [item for item in dataset.train_df.columns.tolist() if item not in dataset.label_columns]
    # print(feature_columns)
    # exit()
    assert sorted(feature_columns) == sorted(dataset.feature_columns)
    metric_name = dataset.metric
    # metric_name = "accuracy"
    test_metric_name = dataset.test_metric if hasattr(dataset, "test_metric") else metric_name

    if metric_name.lower() == "r2":
        # For regression, we use rmse as the evaluation metric, but use r2 for the test metric
        metric_name = "rmse"

    # train_data, tuning_data = train_test_split(
    #     dataset.train_df,
    #     test_size=0.1,
    #     random_state=np.random.RandomState(123)
    # )
    # print("fixed split, no sampling")
    # train_data = dataset.train_df.head(int(0.9 * len(dataset.train_df)))
    # tuning_data = dataset.train_df.tail(int(0.1 * len(dataset.train_df)))

    predictor = AutoMMPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )

    config = {
        MODEL: f"configs/model/{args.model_config}.yaml",
        DATA: "configs/data/default.yaml",
        OPTIMIZATION: "configs/optimization/adamw.yaml",
        ENVIRONMENT: "configs/environment/default.yaml",
    }
    # overrides = ""
    save_path = os.path.join(get_home_dir(), "outputs", args.dataset, args.model_config)
    # for i in range(2):
    predictor.fit(
        train_data=dataset.train_df,
        # tuning_data=dataset.train_df,
        config=config,
        overrides=args.overrides,
        save_path=save_path,
        seed=args.seed,
        init_only=True,
    )
    scores, y_pred = predictor.evaluate(
        data=dataset.test_df,
        metrics=[test_metric_name],
        return_pred=True,
    )
    print(f"before-fit score: {scores}")

    predictor.fit(
        train_data=dataset.train_df,
        # tuning_data=dataset.train_df,
        config=config,
        overrides=args.overrides,
        save_path=save_path,
        seed=args.seed,
    )
    scores, y_pred = predictor.evaluate(
        data=dataset.test_df,
        metrics=[test_metric_name],
        return_pred=True,
    )
    print(f"after-fit score: {scores}")

    with open(os.path.join(predictor.path, 'test_metrics.json'), 'w') as fp:
        json.dump(scores, fp)


if __name__ == '__main__':
    main()
