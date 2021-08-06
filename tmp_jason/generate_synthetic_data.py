import argparse
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--train_path", help="path to training CSV", default="dataset/AdultIncomeBinaryClassification/train_data.csv")
parser.add_argument("-g", "--test_path", help="path to test CSV", default="dataset/AdultIncomeBinaryClassification/test_data.csv")
parser.add_argument('-l', '--label', help='label column name', type=str, default='class')
parser.add_argument('-t', '--times', help='how many synthetic columns to generate based on original feature dimensionality', type=float, default=0.1)
args = parser.parse_args()

# add bunch of Gaussian synthetic features
# add noised versions of original features
train_path = args.train_path
test_path = args.test_path
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
train_len, test_len = len(train_df), len(test_df)
train_path_split = train_path.split("/")
test_path_split = test_path.split("/")

# Create a dataset with a bunch of standard Gaussian synthetic features
num_new_columns = np.ceil(len(train_df.drop(columns=[args.label]).columns) * args.times).astype(int)
train_noises, test_noises = {}, {}
for i in range(num_new_columns):
    train_noise = np.random.normal(loc=0., scale=1., size=train_len)
    test_noise = np.random.normal(loc=0., scale=1., size=test_len)
    train_noises[f"noise_{i+1}"] = train_noise
    test_noises[f"noise_{i+1}"] = test_noise
new_train_df = pd.concat([pd.DataFrame(train_noises), train_df], axis=1)
new_test_df = pd.concat([pd.DataFrame(test_noises), test_df], axis=1)

train_save_path = f"{'/'.join(train_path_split[:-1])}/n1_r{args.times}_{train_path_split[-1]}"
test_save_path = f"{'/'.join(test_path_split[:-1])}/n1_r{args.times}_{test_path_split[-1]}"
new_train_df.to_csv(train_save_path, index=False)
new_test_df.to_csv(test_save_path, index=False)

# Create a dataset with a bunch of features that are shuffled versions
# of the original features
train_noises, test_noises = {}, {}
for i in range(num_new_columns):
    train_noises[f"noise_{i+1}"] = shuffle(train_df[train_df.columns[i % len(train_df.columns)]], random_state=0).tolist()
    test_noises[f"noise_{i+1}"] = shuffle(test_df[test_df.columns[i % len(test_df.columns)]], random_state=0).tolist()
new_train_df = pd.concat([pd.DataFrame(train_noises), train_df], axis=1)
new_test_df = pd.concat([pd.DataFrame(test_noises), test_df], axis=1)

train_save_path = f"{'/'.join(train_path_split[:-1])}/n2_r{args.times}_{train_path_split[-1]}"
test_save_path = f"{'/'.join(test_path_split[:-1])}/n2_r{args.times}_{test_path_split[-1]}"
new_train_df.to_csv(train_save_path, index=False)
new_test_df.to_csv(test_save_path, index=False)
