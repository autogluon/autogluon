import os, csv
import pandas as pd
import numpy as np
from glob import glob

__all__ = ['generate_csv', 'generate_csv_submission', 'generate_prob_csv']

def generate_csv_submission(dataset_path, dataset, local_path, inds, preds, class_name, custom):
    """
    Generate_csv for submission with different formats.
    :param dataset: dataset name.
    :param dataset_path: dataset path.
    :param local_path: save log and plot performance_vs_trials figure.
    :param inds: the category id.
    :param preds: the category probability.
    :param class_name: the category name.
    :param custom: save the name of csv.
    """
    if dataset == 'dogs-vs-cats-redux-kernels-edition':
        csv_config = {'fullname': False,
                      'need_sample': False,
                      'content': 'int',
                      'image_column_name': 'id',
                      'class_column_name': 'label',
                      'value': 'probability_1',
                      'special': 0}
    elif dataset == 'aerial-cactus-identification':
        csv_config = {'fullname': True,
                      'need_sample': False,
                      'image_column_name': 'id',
                      'class_column_name': 'has_cactus',
                      'content': 'empty',
                      'value': 'probability_1',
                      'special': 0}
    elif dataset == 'plant-seedlings-classification':
        csv_config = {'fullname': True,
                      'need_sample': True,
                      'content': 'origin',
                      'value': 'category',
                      'image_column_name': 'file',
                      'class_column_name': 'species',
                      'special': ''
                      }
    elif dataset == 'fisheries_Monitoring':
        csv_config = {'fullname': True,
                      'need_sample': True,
                      'content': 'str',
                      'value': 'multi_prob',
                      'image_column_name': 'image',
                      'class_column_name': '',
                      'special': 'rename'
                      }
    elif dataset == 'dog-breed-identification':
        csv_config = {'fullname': False,
                      'need_sample': True,
                      'content': 'str',
                      'value': 'multi_prob',
                      'image_column_name': 'id',
                      'class_column_name': '',
                      'special': ''
                      }
    elif dataset == 'shopee-iet-machine-learning-competition':
        csv_config = {'fullname': False,
                      'need_sample': False,
                      'image_column_name': 'id',
                      'class_column_name': 'category',
                      'value': 'class_id',
                      'content': 'special',
                      'special': 5}

    elif dataset == 'shopee-iet':# dogs
        csv_config = {'fullname': False,
                      'need_sample': False,
                      'content': 'int',
                      'image_column_name': 'id',
                      'class_column_name': 'label',
                      'value': 'probability_1',
                      'special': 0}

    test_path = os.path.join(dataset_path, 'test')
    csv_path = os.path.join(dataset_path, 'sample_submission.csv')
    ids = sorted(os.listdir(test_path))
    save_csv_name = custom + '.csv'
    save_csv_path = os.path.join(local_path, dataset, save_csv_name)
    if csv_config['need_sample']:  # plant\fish\dog
        df = pd.read_csv(csv_path)
        if not csv_config['fullname']:
            imagename_list = [name_id[:-4] for name_id in ids]
        else:
            imagename_list = ids
        row_index_group = []
        for i in imagename_list:
            if csv_config['content'] == 'str':
                row_index = df[df[csv_config['image_column_name']] == str(i)].index.tolist()
            elif csv_config['content'] == 'origin':
                row_index = df[df[csv_config['image_column_name']] == i].index.tolist()
            elif csv_config['content'] == 'int':
                row_index = df[df[csv_config['image_column_name']] == int(i)].index.tolist()
            elif csv_config['content'] == 'special':
                row_index = df[df[csv_config['image_column_name']] == int(i[5:])].index.tolist()
            row_index_group.append(row_index[0])
        if csv_config['value'] == 'category':
            df.loc[row_index_group, csv_config['class_column_name']] = class_name
        elif csv_config['value'] == 'multi_prob':
            df.loc[row_index_group, 1:] = preds
        if csv_config['special'] == 'rename':
            def get_name(name):
                if name.startswith('image'):
                    name = 'test_stg2/' + name
                return name
            df['image'] = df['image'].apply(get_name)
        df.to_csv(save_csv_path, index=False)
        print('generate_csv A is done')
    else:  # dogs/aerial/shopee
        with open(save_csv_path, 'w') as f:
            row = [csv_config['image_column_name'], csv_config['class_column_name']]
            writer = csv.writer(f)
            writer.writerow(row)
            if csv_config['value'] == 'class_id':
                for i, ind in zip(ids, inds):
                    row = [i, ind]
                    writer = csv.writer(f)
                    writer.writerow(row)
            if csv_config['value'] == 'probability_1':
                for i, prob in zip(ids, preds):
                    row = [i, prob[1]]
                    print(row)
                    writer = csv.writer(f)
                    writer.writerow(row)
            if csv_config['value'] == 'probability_0':
                for i, prob in zip(ids, preds):
                    row = [i, prob[0]]
                    writer = csv.writer(f)
                    writer.writerow(row)
        f.close()
        if not csv_config['fullname']:
            df = pd.read_csv(save_csv_path)
            df['id'] = df['id'].apply(lambda x: int(x[csv_config['special']:-4]))
            df.sort_values("id", inplace=True)
            df.to_csv(save_csv_path, index=False)
        print('generate_csv B is done')

def filter_value(prob, Threshold):
    if prob > Threshold:
        prob = prob
    else:
        prob = 0
    return prob

def generate_prob_csv(test_dataset, preds, set_prob_thresh=0, ensemble_list='', custom='./submission.csv', scale_min_max=True):
    if isinstance(test_dataset.rand, list):
        ids = sorted([x for x, _ in test_dataset.rand[0].items])
        csv_path = test_dataset.rand[0]._root.replace('test', 'sample_submission.csv')
    else:
        ids = sorted([x for x, _ in test_dataset.rand.items])
        csv_path = test_dataset.rand._root.replace('test', 'sample_submission.csv')
    df = pd.read_csv(csv_path)
    imagename_list = [name_id[:-4] for name_id in ids]
    row_index_group = []
    for i in imagename_list:
        row_index = df[df['id'] == str(i)].index.tolist()
        if not len(row_index) == 0:
            row_index_group.append(row_index[0])
    df.loc[row_index_group, 1:] = preds
    df.to_csv(custom, index=False)

    def ensemble_csv(glob_files):
        file_list = []
        for i, glob_file in enumerate(glob(glob_files)):
            file_list.append(pd.read_csv(glob_file, index_col=0))
        w = sum([*file_list])/len(file_list)
        if scale_min_max:
            w = w.apply(lambda x: np.round((x - min(x)) / (1.0 * (max(x) - min(x))), 2), axis=1)
        for i in w.columns.values:
            w[i] = w[i].apply(filter_value, Threshold=set_prob_thresh)
        w.to_csv(custom)
    if not ensemble_list.strip() == '':
        ensemble_csv(ensemble_list)
    print('dog_generate_csv is done')

def generate_csv(inds, path):
    with open(path, 'w') as csvFile:
        row = ['id', 'category']
        writer = csv.writer(csvFile)
        writer.writerow(row)
        id = 1
        for ind in inds:
            row = [id, ind]
            writer = csv.writer(csvFile)
            writer.writerow(row)
            id += 1
    csvFile.close()
