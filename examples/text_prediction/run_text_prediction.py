import mxnet as mx
import os
import json
import argparse
import warnings
from autogluon.task.text_prediction import TextPrediction
mx.npx.set_np()

warnings.filterwarnings("ignore", category=DeprecationWarning)


TASKS = \
    {'cola': ('sentence', 'label', 'mcc', ['mcc']),
     'sst': ('sentence', 'label', 'acc', ['acc']),
     'mrpc': (['sentence1', 'sentence2'], 'label', 'f1', ['acc', 'f1']),
     'sts': (['sentence1', 'sentence2'], 'score', 'spearmanr', ['pearsonr', 'spearmanr']),
     'qqp': (['sentence1', 'sentence2'], 'label', 'f1', ['acc', 'f1']),
     'mnli': (['sentence1', 'sentence2'], 'label', 'acc', ['acc']),
     'qnli': (['question', 'sentence'], 'label', 'acc', ['acc']),
     'rte': (['sentence1', 'sentence2'], 'label', 'acc', ['acc']),
     'wnli': (['sentence1', 'sentence2'], 'label', 'acc', ['acc']),
     'snli': (['sentence1', 'sentence2'], 'label', 'acc', ['acc'])}


def parse_args():
    parser = argparse.ArgumentParser(description='AutoML for Tabular Prediction Basic Example.')
    parser.add_argument('--train_file', type=str,
                        help='The training pandas dataframe.',
                        default=None)
    parser.add_argument('--dev_file', type=str,
                        help='The validation pandas dataframe',
                        default=None)
    parser.add_argument('--test_file', type=str,
                        help='The test pandas dataframe',
                        default=None)
    parser.add_argument('--seed', type=int, help='The seed',
                        default=None)
    parser.add_argument('--task', type=str,
                        help='The default tasks',
                        default=None)
    parser.add_argument('--save_dir', type=str,
                        help='The directory to save the experiment',
                        default=None)
    parser.add_argument('--eval_metrics', type=str,
                        help='The metrics for evaluating the models.',
                        default=None)
    parser.add_argument('--stop_metric', type=str,
                        help='The metrics for early stopping',
                        default=None)
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to train the model')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to evaluate the model')
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='The experiment directory where the model params will be written.')
    parser.add_argument('--config_file', type=str, default=None)
    args = parser.parse_args()
    return args


def train(args):
    if args.task is not None:
        feature_columns, label_columns, stop_metric, eval_metrics = TASKS[args.task]
    else:
        raise NotImplementedError
    if args.exp_dir is None:
        args.exp_dir = 'autogluon_{}'.format(args.task)
    task = TextPrediction()
    model = task.fit(train_data=args.train_file,
                     label=label_columns,
                     feature_columns=feature_columns,
                     output_directory=args.exp_dir,
                     stopping_metric=stop_metric,
                     nthreads_per_trial=4,
                     ngpus_per_trial=1,
                     eval_metric=eval_metrics)
    dev_metrics_scores = model.evaluate(args.dev_file, metrics=eval_metrics)
    with open(os.path.join(args.exp_dir, 'final_model_dev_score.json'), 'w') as of:
        json.dump(dev_metrics_scores, of)
    dev_prediction = model.predict(args.dev_file)
    test_prediction = model.predict(args.test_file)
    with open(os.path.join(args.exp_dir, 'dev_predictions.txt'), 'w') as of:
        for ele in dev_prediction:
            of.write(str(ele) + '\n')
    with open(os.path.join(args.exp_dir, 'test_predictions.txt'), 'w') as of:
        for ele in test_prediction:
            of.write(str(ele) + '\n')
    model.save(os.path.join(args.exp_dir, 'saved_model'))


def predict(args):
    raise NotImplementedError


if __name__ == '__main__':
    args = parse_args()
    if args.do_train:
        train(args)
    if args.do_eval:
        predict(args)
