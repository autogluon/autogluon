from autogluon.text.automm import AutoMMPredictor
import argparse
from dataset import (
    AdultTabularDataset
)

TABULAR_DATASETS = {
    'ad': AdultTabularDataset,
}

hyperparameters = {
    'data.categorical.convert_to_text': False,
    'model.names': ["categorical_transformer","numerical_transformer","fusion_transformer"],
    'env.batch_size': 512,
    'env.per_gpu_batch_size': 512,
    'env.num_workers': 12,
    'env.num_workers_evaluation': 12,
    'optimization.max_epochs': 1000,
    'optimization.weight_decay': 1.0e-5,
    'env.num_gpus': 1,
}

def main(args):
    assert args.dataset_name in TABULAR_DATASETS.keys()
    
    ### Dataset loading
    train_data = TABULAR_DATASETS[args.dataset_name](
        'train',args.dataset_dir
    )
    
    val_data = TABULAR_DATASETS[args.dataset_name](
        'val',args.dataset_dir
    )
    
    test_data = TABULAR_DATASETS[args.dataset_name](
        'test',args.dataset_dir
    )
    
    ### model initalization
    predictor = AutoMMPredictor(
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
        hyperparameters=hyperparameters
    )

    ### model inference
    predictor.evaluate(
        data=test_data.data,
        metrics=[test_data.metric]
    )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ad', type=str)
    parser.add_argument('--dataset_dir', default='/home/ubuntu/dataset', type=str)
    parser.add_argument('--exp_dir', default='/home/ubuntu/result/test', type=str)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    main(args)
