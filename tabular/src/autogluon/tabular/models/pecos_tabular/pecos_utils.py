import pandas as pd
import json
import re
import numpy as np


def clean_str(s):
    """
    Convert all non-alphanumeric (or _) chars to _
    """
    return re.sub('\W+', '_', str(s))


def read_pred_outfile(pred_outfile, k = 1):
    """
    Reads predictions output from PECOS
    :param pred_outfile --> PECOS prediction output
    :param k --> number of predictions to output for each example
    :return Dataframe where each row contains maps to a datapoint with format:
            [(pred_1, pred_2, ..., pred_k,), (confidence_1, confidence_2, ... confidence_k,)
    """
    df_pred = pd.DataFrame(
        [
            (
                tuple(t[0] for t in r['data'][:k]),
                tuple(t[1] for t in r['data'][:k])
            )
            for r in load_json_multi(pred_outfile)
        ],
        columns=['labels', 'scores']
    )
    return df_pred 


def load_json_multi(fn):
    with open(fn) as f:
        for ln in f:
            yield json.loads(ln)


def format_predictions(df_pred):
    """
    Get an np array of top predictions and corresponding confidence
    Necessary formatting for AutoGluon
    """
    y_pred = []
    confidence = []
    for i,row in df_pred.iterrows():
        y_pred.append(row['labels'][0])
        confidence.append(row['scores'][0])
    return np.array(y_pred), np.array(confidence)
