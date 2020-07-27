import os
import argparse
import pandas as pd
import shutil
import tarfile
from autogluon_contrib_nlp.utils.misc import download, load_checksum_stats
from autogluon_contrib_nlp.base import get_data_home_dir, get_repo_url


TASK2PATH = {
    "ag": get_repo_url() + "datasets/text_classification/ag_news_csv.tar.gz",
    "imdb": get_repo_url() + "datasets/text_classification/imdb.tar.gz",
    "dbpedia": get_repo_url() + "datasets/text_classification/dbpedia_csv.tar.gz",
    "yelp2": get_repo_url() + "datasets/text_classification/yelp_review_polarity_csv.tar.gz",
    "yelp5": get_repo_url() + "datasets/text_classification/yelp_review_full_csv.tar.gz",
    "amazon2": get_repo_url() + "datasets/text_classification/amazon_review_polarity_csv.tar.gz",
    "amazon5": get_repo_url() + "datasets/text_classification/amazon_review_full_csv.tar.gz",
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tasks",
                        help="tasks to download data for as a comma separated string",
                        type=str,
                        default="all")
    parser.add_argument("-d", "--data_dir",
                        help="Directory to save data to", type=str,
                        default='text_classification_benchmark')
    parser.add_argument('--cache-path', type=str,
                        default=os.path.join(get_data_home_dir(), 'text_classification'),
                        help='The temporary path to download the dataset.')
    return parser


def main(args):
    os.makedirs(args.cache_path, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    if args.tasks == 'all':
        tasks = list(TASK2PATH.keys())
    else:
        tasks = args.tasks.split(',')
    for task in tasks:
        task_dir_path = os.path.join(args.data_dir, task)
        os.makedirs(task_dir_path, exist_ok=True)
        file_url = TASK2PATH[task]
        download_path = download(file_url, args.cache_path)
        with tarfile.open(download_path) as f:
            f.extractall(task_dir_path)
        if task == 'imdb':
            shutil.move(os.path.join(task_dir_path, 'imdb', 'train.parquet'),
                        os.path.join(task_dir_path, 'train.parquet'))
            shutil.move(os.path.join(task_dir_path, 'imdb', 'test.parquet'),
                        os.path.join(task_dir_path, 'test.parquet'))
            train_data = pd.read_parquet(os.path.join(task_dir_path, 'train.parquet'))
            test_data = pd.read_parquet(os.path.join(task_dir_path, 'test.parquet'))
            print('Task={}, #Train={}, #Test={}'.format(task, len(train_data), len(test_data)))
        elif task == 'ag':
            train_data = pd.read_csv(os.path.join(task_dir_path, 'ag_news_csv', 'train.csv'),
                                     header=None)
            test_data = pd.read_csv(os.path.join(task_dir_path, 'ag_news_csv', 'test.csv'),
                                    header=None)
            train_data = pd.DataFrame({'label': train_data[0],
                                       'content': train_data[1] + ' ' + train_data[2]})
            test_data = pd.DataFrame({'label': test_data[0],
                                      'content': test_data[1] + ' ' + test_data[2]})
            train_data.to_parquet(os.path.join(task_dir_path, 'train.parquet'))
            test_data.to_parquet(os.path.join(task_dir_path, 'test.parquet'))
            print('Task={}, #Train={}, #Test={}'.format(task, len(train_data), len(test_data)))
        elif task == 'dbpedia':
            train_data = pd.read_csv(os.path.join(task_dir_path, 'dbpedia_csv', 'train.csv'),
                                     header=None)
            test_data = pd.read_csv(os.path.join(task_dir_path, 'dbpedia_csv', 'test.csv'),
                                    header=None)
            train_data = pd.DataFrame({'label': train_data[0],
                                       'content': train_data[1] + ' ' + train_data[2]})
            test_data = pd.DataFrame({'label': test_data[0],
                                      'content': test_data[1] + ' ' + test_data[2]})
            train_data.to_parquet(os.path.join(task_dir_path, 'train.parquet'))
            test_data.to_parquet(os.path.join(task_dir_path, 'test.parquet'))
            print('Task={}, #Train={}, #Test={}'.format(task, len(train_data), len(test_data)))
        elif task == 'yelp2':
            train_data = pd.read_csv(os.path.join(task_dir_path, 'yelp_review_polarity_csv',
                                                  'train.csv'), header=None)
            test_data = pd.read_csv(os.path.join(task_dir_path, 'yelp_review_polarity_csv',
                                                 'test.csv'), header=None)
            train_data.columns = ['label', 'review']
            test_data.columns = ['label', 'review']
            train_data.to_parquet(os.path.join(task_dir_path, 'train.parquet'))
            test_data.to_parquet(os.path.join(task_dir_path, 'test.parquet'))
        elif task == 'yelp5':
            train_data = pd.read_csv(os.path.join(task_dir_path,
                                                  'yelp_review_full_csv',
                                                  'train.csv'), header=None)
            test_data = pd.read_csv(os.path.join(task_dir_path,
                                                 'yelp_review_full_csv',
                                                 'test.csv'), header=None)
            train_data.columns = ['label', 'review']
            test_data.columns = ['label', 'review']
            train_data.to_parquet(os.path.join(task_dir_path, 'train.parquet'))
            test_data.to_parquet(os.path.join(task_dir_path, 'test.parquet'))
        elif task == 'amazon2':
            train_data = pd.read_csv(os.path.join(task_dir_path,
                                                  'amazon_review_polarity_csv',
                                                  'train.csv'), header=None)
            test_data = pd.read_csv(os.path.join(task_dir_path,
                                                 'amazon_review_polarity_csv',
                                                 'test.csv'), header=None)
            train_data = pd.DataFrame({'label': train_data[0],
                                       'review': train_data[1] + ' ' + train_data[2]})
            test_data = pd.DataFrame({'label': test_data[0],
                                      'review': test_data[1] + ' ' + test_data[2]})
            train_data.to_parquet(os.path.join(task_dir_path, 'train.parquet'))
            test_data.to_parquet(os.path.join(task_dir_path, 'test.parquet'))
        elif task == 'amazon5':
            train_data = pd.read_csv(os.path.join(task_dir_path,
                                                  'amazon_review_full_csv',
                                                  'train.csv'), header=None)
            test_data = pd.read_csv(os.path.join(task_dir_path,
                                                 'amazon_review_full_csv',
                                                 'test.csv'), header=None)
            train_data = pd.DataFrame({'label': train_data[0],
                                       'review': train_data[1] + ' ' + train_data[2]})
            test_data = pd.DataFrame({'label': test_data[0],
                                      'review': test_data[1] + ' ' + test_data[2]})
            train_data.to_parquet(os.path.join(task_dir_path, 'train.parquet'))
            test_data.to_parquet(os.path.join(task_dir_path, 'test.parquet'))
        else:
            raise NotImplementedError


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
