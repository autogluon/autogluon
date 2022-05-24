from autogluon.text.automm import AutoMMPredictor
import argparse
from utils import get_tabular_data 

def main(args):

    train_df, val_df, test_df, column_types, problem_type, metric = get_tabular_data(
        base_path=args.dataset_dir,
        dataset_name=args.dataset_name,
    )

    predictor = AutoMMPredictor(
        label='0', 
        problem_type=problem_type,
        eval_metric=metric,
        path=args.exp_dir,
        verbosity=4,
    )

    predictor.fit(
        train_data=train_df,
        tuning_data=test_df,
        column_types=column_types,
        seed=args.seed,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='./data', type=str)
    parser.add_argument('--dataset_name', default='ad', type=str)
    parser.add_argument('--exp_dir', default='./result/', type=str)
    parser.add_argument('--seed', default=95, type=int)

    args = parser.parse_args()

    main(args)