import argparse
import re

import gluonnlp as nlp
import numpy as np
import pandas as pd


# run `sh download_quora_question_pairs.sh` first

def train_validate_test_split(df, split=.9, seed=7):
    np.random.seed(seed)
    perm = np.random.permutation(len(df))
    m = len(df)
    train_end = int(split * m)
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:]]
    return train, validate


parser = argparse.ArgumentParser(description='Prepare Quora Question Pairs dataset')
parser.add_argument('--data', type=str, required=True, help='path to the folder where the training file is downloaded.')
parser.add_argument('--split', type=float, default=0.9, help='The training and validation dataset split')


def clean_text(data_frame):
    def clean(text):
        text = str(text).lower()
        text = re.sub(u'\n', '', text)
        # More cleaning functions such as replacing HTML URLs, numbers etc can be added here.
        return text

    data_frame['question1'] = data_frame['question1'].apply(lambda x: clean(x))
    data_frame['question2'] = data_frame['question2'].apply(lambda x: clean(x))


if __name__ == '__main__':
    args = parser.parse_args()

    df = pd.read_csv('{}/train.csv'.format(args.data))

    print(df.head())

    clean_text(df)

    train_df, valid_df = train_validate_test_split(df, split=args.split)

    train_df.to_csv('{}/train.tsv'.format(args.data), sep='\t', index=None)
    valid_df.to_csv('{}/val.tsv'.format(args.data), sep='\t', index=None)

    # Validate whether our steps were correct or not.

    train_data = nlp.data.TSVDataset(filename='{}/train.tsv'.format(args.data), num_discard_samples=1,
                                     field_indices=[3, 4, 5])
    valid_data = nlp.data.TSVDataset(filename='{}/val.tsv'.format(args.data), num_discard_samples=1,
                                     field_indices=[3, 4, 5])

    import os

    os.remove('{}/train.csv'.format(args.data))

    assert (len(train_df) == len(train_data))
    assert (len(valid_df) == len(valid_data))

    print('DONE!')
