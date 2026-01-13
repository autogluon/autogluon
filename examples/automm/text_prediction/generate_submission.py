import argparse
import os

import numpy as np
import pandas as pd

from autogluon.multimodal import MultiModalPredictor


def get_parser():
    parser = argparse.ArgumentParser("Generate GLUE submission folder")
    parser.add_argument("--prefix", type=str, default="autogluon_text")
    parser.add_argument("--save_dir", type=str, default="glue_submission")
    return parser


def get_test_index(path):
    with open(path, "r", encoding="utf-8") as in_f:
        lines = in_f.readlines()
    index_l = []
    for i in range(1, len(lines)):
        index_l.append(lines[i].split()[0])
    return index_l


def main(args):
    tasks = {
        "cola": ["CoLA.tsv", "glue/cola/test.tsv"],
        "sst": ["SST-2.tsv", "glue/sst/test.tsv"],
        "mrpc": ["MRPC.tsv", "glue/mrpc/test.tsv"],
        "sts": ["STS-B.tsv", "glue/sts/test.tsv"],
        "qqp": ["QQP.tsv", "glue/qqp/test.tsv"],
        "mnli_m": ["MNLI-m.tsv", "glue/mnli/test_matched.tsv"],
        "mnli_mm": ["MNLI-mm.tsv", "glue/mnli/test_mismatched.tsv"],
        "qnli": ["QNLI.tsv", "glue/qnli/test.tsv"],
        "rte": ["RTE.tsv", "glue/rte/test.tsv"],
        "wnli": ["WNLI.tsv", "glue/wnli/test.tsv"],
        "ax": ["AX.tsv", "glue/rte_diagnostic/diagnostic.tsv"],
    }

    os.makedirs(args.save_dir, exist_ok=True)

    for task, (save_name, test_file_path) in tasks.items():
        if task == "ax":
            # For AX, we need to load the mnli-m checkpoint and run inference
            test_df = pd.read_csv(test_file_path, sep="\t", header=0)
            test_index = test_df["index"]
            predictor = MultiModalPredictor.load(f"{args.prefix}_mnli_m")
            label_column = predictor.label
            predictions = predictor.predict(test_df)
        else:
            test_index = get_test_index(test_file_path)
            prediction_df = pd.read_csv(f"{args.prefix}_{task}/test_prediction.csv", index_col=0)
            label_column = prediction_df.columns[0]
            predictions = prediction_df[label_column]
        if task == "sts":
            predictions = np.clip(predictions, 0, 5)
        with open(os.path.join(args.save_dir, save_name), "w") as of:
            of.write("index\t{}\n".format(label_column))
            for i in range(len(predictions)):
                of.write("{}\t{}\n".format(test_index[i], predictions[i]))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
