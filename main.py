import numpy as np
import pandas as pd

from pyDeepInsight import ImageTransformer
from sklearn.model_selection import train_test_split

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # load data
    logging.info("loading data")
    expr_file = r"./data/tcga.rnaseq_fpkm_uq.example.txt.gz"
    expr = pd.read_csv(expr_file, sep="\t")
    y = expr['project'].values
    X = expr.iloc[:, 1:].values
    logging.info("making train/test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, stratify=y)
    logging.info("initializing IT")
    it = ImageTransformer(n_jobs=-1)
    logging.info("training IT and transforming training data")
    mat_train = it.fit_transform(X_train)
    logging.info("transforming test data")
    mat_test = it.transform(X_test)
    logging.info("done")
