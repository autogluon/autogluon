import pandas as pd
from autogluon.text import TextPredictor
import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser('Generate GLUE submission folder')
    parser.add_argument('--prefix', type=str, default='autogluon_text')
    parser.add_argument('--save_dir', type=str, default='glue_submission')
    return parser


def main(args):
    tasks = {
        'cola':    ['CoLA.tsv',  'glue/cola/test.tsv'],
        'sst':     ['SST-2.tsv', 'glue/sst/test.tsv'],
        'mrpc':    ['MRPC.tsv',  'glue/mrpc/test.tsv'],
        'sts':     ['STS-B.tsv', 'glue/sts/test.tsv'],
        'qqp':     ['QQP.tsv', 'glue/qqp/test.tsv'],
        'mnli-m':  ['MNLI-m.tsv', 'glue/mnli/test_matched.tsv'],
        'mnli-mm': ['MNLI-mm.tsv', 'glue/mnli/test_mismatched.tsv'],
        'qnli':    ['QNLI.tsv', 'glue/qnli/test.tsv'],
        'rte':     ['RTE.tsv', 'glue/rte/test.tsv'],
        'wnli':    ['WNLI.tsv', 'glue/wnli/test.tsv'],
        'ax':      ['AX.tsv', 'glue/rte_diagnostic/diagnostic.tsv']
    }

    os.makedirs(args.save_dir, exist_ok=True)

    for task, (save_name, test_file_path) in tasks.items():
        print('Load {}'.format(test_file_path))
        test_df = pd.read_csv(test_file_path, sep='\t', header=0)
        if task == 'ax':
            # For AX, we need to load the mnli-m checkpoint and run inference
            predictor = TextPredictor.load(f'{args.prefix}_mnli_m')
            label_column = predictor.label
            predictions = predictor.predict(test_df)
        else:
            prediction_df = pd.read_csv(f'{args.prefix}_{task}/test_prediction.csv',
                                        index_col=0)
            label_column = prediction_df.columns[0]
            predictions = prediction_df[label_column]
        with open(os.path.join(args.save_dir, save_name), 'w') as of:
            of.write('index\t{}\n'.format(label_column))
            for i in range(len(predictions)):
                of.write('{}\t{}\n'.format(test_df['index'][i],
                                           predictions[i]))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
