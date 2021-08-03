import argparse
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.models import CatBoostModel, KNNModel, LGBModel
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--train_path', help='train dataset path', type=str, default='dataset/australian/train_data.csv')
parser.add_argument('-g', '--test_path', help='train dataset path', type=str, default='dataset/australian/test_data.csv')
parser.add_argument('-m', '--mode', help='what AutoGluon setting to try', choices=['model', 'model-stack', 'ag', 'ag-stack'], default='model')
parser.add_argument('-n', '--num_resource', help='number of resource to allocate across all features', type=int, default=None)
parser.add_argument('-l', '--label', help='name of the label column', type=str, default='class')
parser.add_argument('-p', '--prune', help='to use fit_with_prune or not', dest='prune', action='store_true')
parser.add_argument('-r', '--ratio', help='what percentage of features to prune at once', type=float, default=0.1)
parser.add_argument('-x', '--fi', help='feature importance strategy', default='uniform', choices=['uniform', 'backwardsearch'])
parser.add_argument('-y', '--fp', help='feature pruning strategy or not', default='percentage', choices=['percentage', 'single'])
args = parser.parse_args()

train_data = pd.read_csv(args.train_path).head(10000)
test_data = pd.read_csv(args.test_path)
X_test = test_data.drop(columns=[args.label])
y_test = test_data[args.label]

fit_with_prune_kwargs = {
    'fit_with_prune_kwargs': {
        'max_num_fit': 3,
        'stop_threshold': 1,
        'prune_ratio': args.ratio,
        'prune_threshold': 0.,
        'subsample_size': 5000,
        'num_resource': args.num_resource,
        'fi_strategy': args.fi,
        'fp_strategy': args.fp,
        # 'prune_after_fit': False
    }
}

if args.mode == 'model':
    presets = ['medium_quality_faster_train']
    custom_hyperparameters = {LGBModel: {}}
elif args.mode == 'model-stack':
    presets = ['best_quality']
    # custom_hyperparameters = {KNNModel: {'ag_args': fit_with_prune_kwargs}}
    custom_hyperparameters = {KNNModel: {}, CatBoostModel: {}}
    # custom_hyperparameters = {CatBoostModel: {}}
elif args.mode == 'ag':
    presets = ['medium_quality_faster_train']
    custom_hyperparameters = None
else:
    presets = ['best_quality']
    custom_hyperparameters = None

predictor = TabularPredictor(label=args.label)
if args.prune:
    predictor = predictor.fit(train_data, presets=presets, ag_args=fit_with_prune_kwargs, time_limit=300, num_bag_sets=2, num_stack_levels=1,
                              ag_args_ensemble={'use_child_oof': False}, hyperparameters=custom_hyperparameters)
else:
    predictor = predictor.fit(train_data, presets=presets, # num_bag_sets=2, num_stack_levels=1, 
                              time_limit=900, hyperparameters=custom_hyperparameters)
# ag_args=fit_with_prune_kwargs
try:
    y_pred = predictor.predict(test_data)
    performance = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    print(performance)
except:
    import pdb; pdb.post_mortem()
# import pdb; pdb.set_trace()
# result = predictor.feature_importance(data=test_data, model='KNeighbors_BAG_L2', features=['sex'])
# print(result)
# import pdb; pdb.set_trace()
# result = predictor.feature_importance(data=test_data, model='KNeighbors_BAG_L2', features=['sex'])
# print(result)
