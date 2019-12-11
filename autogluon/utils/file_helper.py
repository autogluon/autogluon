import os, glob, csv
import pandas as pd

__all__ = ['generate_csv', 'generate_csv_submission']

def generate_csv_submission(csv_path, dataset, load_dataset, classifier, local_path, csv_config = None):
    """
    Generate_csv for submission :
    save predict_2.csv for submission with different formats.
    :param csv_path: save predict_2.csv file
    :param dataset: the dataset's name
    :param load_dataset: task.Dataset Object
    :param classifier: fit Object #
    :param local_path: script path
    :param csv_config: submission config params
    """
    if dataset == 'dogs-vs-cats-redux-kernels-edition/':
        """ submit a probability that image is a dog """
        csv_config = {'fullname' : False,
                       'output': 'ab',# multi/oneot/ab
                       'image_column_name' : 'id',
                       'content' : 'int',
                       'class_column_name' : 'label',
                       'style': 'jpg',
                       'value' : 'probability' # category,
                       }
    elif dataset == 'aerial-cactus-identification/':
        """ predict a probability for the has_cactus variable """
        csv_config = {'fullname' : True,
                       'output': 'ab',
                       'image_column_name' : 'id',
                       'content': 'empty',
                       'class_column_name' : 'has_cactus',
                       'style': 'jpg',
                       'value' : 'probability'
                       }
    elif dataset == 'plant-seedlings-classification/':
        """ MeanF1Score """
        csv_config = {'fullname' : True,
                       'output': 'ab',
                       'image_column_name' : 'file',
                       'content': 'empty',
                       'class_column_name' : 'species',
                       'style': 'png',
                       'value': 'category'
                       }
    elif dataset == 'fisheries_Monitoring/':
        """ multi-class logarithmic loss """
        """submit a csv file with the image file name, and a probability for each class. """
        csv_config = {'fullname' : True,
                       'output': 'multi',
                       'image_column_name' : 'image',
                       'content': 'str',
                       'class_column_name' : '',
                       'style': 'jpg',
                       'value': 'multi'
                       }
    elif dataset == 'dog-breed-identification/':
        """ multi-class logarithmic loss """
        """ you must predict a probability for each of the different breeds."""
        csv_config = {'fullname' : False,
                       'output': 'multi',
                       'image_column_name' : 'id',
                       'content': 'str',
                       'class_column_name' : '',
                       'style': 'jpg',
                       'value': 'multi'
                       }
    elif dataset == 'shopee-iet-machine-learning-competition/':
        """ """
        csv_config = {'fullname' : False,
                       'output': 'multi',
                       'image_column_name' : 'id',
                       'content': 'special',
                       'class_column_name' : 'category',
                       'style': 'jpg',
                       'value': 'index'
                       }
    elif dataset == 'shopee-iet/':
        csv_config = {'fullname' : False,
                       'output': 'multi',
                       'image_column_name' : 'id',
                       'content': 'special',
                       'class_column_name' : 'category',
                       'style': 'jpg',
                       'value' : 'index'
                       }

    test_path = csv_path.replace('sample_submission.csv', 'test')
    imgs = glob.glob(test_path + "/*." + csv_config['style'])
    df = pd.read_csv(csv_path)
    target_dataset = load_dataset.init()
    for imgname in imgs:
        midname = imgname[imgname.rindex("/") + 1:]
        start_path = os.path.join(test_path, midname)
        ind, prob, prob_all = classifier.predict(start_path, plot=False)
        if not csv_config['fullname']:
            midname = midname[:-4]
        if csv_config['content'] == 'str':
            row_index = df[df[csv_config['image_column_name']] == str(midname)].index.tolist()
        elif csv_config['content'] == 'empty':
            row_index = df[df[csv_config['image_column_name']] == midname].index.tolist()
        elif csv_config['content'] == 'int':
            row_index = df[df[csv_config['image_column_name']] == int(midname)].index.tolist()
        elif csv_config['content'] == 'special':
            row_index = df[df[csv_config['image_column_name']] == int(midname[5:])].index.tolist()
        if csv_config['value'] == 'category':
            value = target_dataset.classes[ind.asscalar()]
        elif csv_config['value'] == 'probability': # fix
            # value = prob.asscalar() if ind.asscalar()== 0 else 1 - prob.asscalar() #
            value = prob.asscalar() if ind.asscalar()== 1 else 1 - prob.asscalar() # cat/dog
        elif csv_config['value'] == 'multi':# fix
            value = prob_all # cat/dog
        else: #index # fix
            value = ind.asscalar()
        if csv_config['output'] == 'ab':
            df.loc[int(row_index[0]), csv_config['class_column_name']] = value
        elif csv_config['output'] == 'multi':
            for i in range(prob_all.shape[1]):# (1,n)
                dd = prob_all[0, i].asscalar()
                value = target_dataset.classes[i]
                df.loc[int(row_index[0]), value] = dd
        else:
            df.loc[int(row_index[0]), 1:] = 0# onehot
            df.loc[int(row_index[0]), value] = 1
    df.to_csv(os.path.join(local_path, dataset, 'predict.csv'), index=False)
    print('predict.csv is done')

def generate_csv(inds, path):
    with open(path, 'w') as csvFile:
        row = ['id', 'category']
        writer = csv.writer(csvFile)
        writer.writerow(row)
        id = 1
        for ind in inds:
            row = [id, ind.asscalar()]
            writer = csv.writer(csvFile)
            writer.writerow(row)
            id += 1
    csvFile.close()
