# Disclaimer! The script here is partially based on
# https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py
# and
# https://github.com/nyu-mll/jiant/blob/master/scripts/download_superglue_data.py
import argparse
import json
import os
import shutil
import zipfile

import pandas as pd
import pyarrow
import pyarrow.json
from autogluon_contrib_nlp.base import get_data_home_dir
from autogluon_contrib_nlp.data.tokenizers import WhitespaceTokenizer
from autogluon_contrib_nlp.utils.misc import download, load_checksum_stats

_CITATIONS = """
@inproceedings{wang2019glue,
  title={GLUE: A multi-task benchmark and analysis platform for natural language understanding},
  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R},
  booktitle={ICLR},
  year={2019}
}

@inproceedings{wang2019superglue,
  title={Superglue: A stickier benchmark for general-purpose language understanding systems},
  author={Wang, Alex and Pruksachatkun, Yada and Nangia, Nikita and Singh, Amanpreet and 
  Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3261--3275},
  year={2019}
}
"""

GLUE_TASKS = ["cola", "sst", "mrpc", "qqp", "sts", "mnli", "snli", "qnli", "rte", "wnli", "diagnostic"]
SUPERGLUE_TASKS = [
    "cb",
    "copa",
    "multirc",
    "rte",
    "wic",
    "wsc",
    "boolq",
    "record",
    "broadcoverage-diagnostic",
    "winogender-diagnostic",
]

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_URL_FILE_STATS = load_checksum_stats(os.path.join(_CURR_DIR, "url_checksums", "glue.txt"))
_URL_FILE_STATS.update(load_checksum_stats(os.path.join(_CURR_DIR, "url_checksums", "superglue.txt")))


def read_tsv_glue(tsv_file, num_skip=1, keep_column_names=False):
    out = []
    nrows = None
    if keep_column_names:
        assert num_skip == 1
    column_names = None
    with open(tsv_file, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i < num_skip:
                if keep_column_names:
                    column_names = line.split()
                continue
            elements = line.split("\t")
            out.append(elements)
            if nrows is None:
                nrows = len(elements)
            else:
                assert nrows == len(elements)
    df = pd.DataFrame(out, columns=column_names)
    series_l = []
    for col_name in df.columns:
        idx = df[col_name].first_valid_index()
        val = df[col_name][idx]
        if isinstance(val, str):
            try:
                dat = pd.to_numeric(df[col_name])
                series_l.append(dat)
                continue
            except ValueError:
                pass
            finally:
                pass
        series_l.append(df[col_name])
    new_df = pd.DataFrame({name: series for name, series in zip(df.columns, series_l)})
    return new_df


def read_jsonl_superglue(jsonl_file):
    columns = None
    out = []
    with open(jsonl_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            sample = json.loads(line)
            if columns is None:
                columns = list(sample.keys())
            else:
                assert sorted(columns) == sorted(list(sample.keys())), "Columns={}, sample.keys()={}".format(
                    columns, sample.keys()
                )
            out.append([sample[col] for col in columns])
    df = pd.DataFrame(out, columns=columns)
    return df


# Classification will be stored as pandas dataframe
def read_cola(dir_path):
    df_dict = dict()
    for fold in ["train", "dev", "test"]:
        csv_file = os.path.join(dir_path, "{}.tsv".format(fold))
        if fold == "test":
            df = pd.read_csv(csv_file, "\t")
            df = df[["sentence"]]
            df_dict[fold] = df
        else:
            df = pd.read_csv(csv_file, "\t", header=None)
            df = df[[3, 1]]
            df.columns = ["sentence", "label"]
            df_dict[fold] = df
    return df_dict, None


def read_sst(dir_path):
    df_dict = dict()
    for fold in ["train", "dev", "test"]:
        csv_file = os.path.join(dir_path, "{}.tsv".format(fold))
        df = pd.read_csv(csv_file, "\t")
        if fold == "test":
            df = df[["sentence"]]
        df_dict[fold] = df
    return df_dict, None


def read_mrpc(dir_path):
    df_dict = dict()
    for fold in ["train", "dev", "test"]:
        tsv_file = os.path.join(dir_path, "{}.tsv".format(fold))
        df = read_tsv_glue(tsv_file)
        if fold == "test":
            df = df[[3, 4]]
            df.columns = ["sentence1", "sentence2"]
        else:
            df = df[[3, 4, 0]]
            df.columns = ["sentence1", "sentence2", "label"]
        df_dict[fold] = df
    return df_dict, None


def read_qqp(dir_path):
    df_dict = dict()
    for fold in ["train", "dev", "test"]:
        csv_file = os.path.join(dir_path, "{}.tsv".format(fold))
        df = pd.read_csv(csv_file, "\t")
        if fold == "test":
            df = df[["question1", "question2"]]
            df.columns = ["sentence1", "sentence2"]
        else:
            df = df[["question1", "question2", "is_duplicate"]]
            df.columns = ["sentence1", "sentence2", "label"]
        df_dict[fold] = df
    return df_dict, None


def read_sts(dir_path):
    df_dict = dict()
    for fold in ["train", "dev", "test"]:
        csv_file = os.path.join(dir_path, "{}.tsv".format(fold))
        df = read_tsv_glue(csv_file)
        if fold == "test":
            df = df[[7, 8, 1]]
            df.columns = ["sentence1", "sentence2", "genre"]
        else:
            df = df[[7, 8, 1, 9]]
            df.columns = ["sentence1", "sentence2", "genre", "score"]
        genre_l = []
        for ele in df["genre"].tolist():
            if ele == "main-forum":
                genre_l.append("main-forums")
            else:
                genre_l.append(ele)
        df["genre"] = pd.Series(genre_l)
        df_dict[fold] = df
    return df_dict, None


def read_mnli(dir_path):
    df_dict = dict()
    for fold in ["train", "dev_matched", "dev_mismatched", "test_matched", "test_mismatched"]:
        csv_file = os.path.join(dir_path, "{}.tsv".format(fold))
        df = read_tsv_glue(csv_file, 1, True)
        if "test" in fold:
            df = df[["sentence1", "sentence2", "genre"]]
        else:
            df = df[["sentence1", "sentence2", "genre", "gold_label"]]
            df.columns = ["sentence1", "sentence2", "genre", "label"]
        df_dict[fold] = df
    return df_dict, None


def read_snli(dir_path):
    df_dict = dict()
    for fold in ["train", "dev", "test"]:
        csv_file = os.path.join(dir_path, "{}.tsv".format(fold))
        column_names = None
        out = []
        with open(csv_file) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0:
                    column_names = line.split()
                    column_names = column_names[:10] + [column_names[-1]]
                    continue
                elements = line.split("\t")
                first_few_elements = elements[:10]
                gold_label = elements[-1]
                out.append(first_few_elements + [gold_label])
        df = pd.DataFrame(out, columns=column_names)
        df = df[["sentence1", "sentence2", "gold_label"]]
        df.columns = ["sentence1", "sentence2", "label"]
        df_dict[fold] = df
    return df_dict, None


def read_qnli(dir_path):
    df_dict = dict()
    for fold in ["train", "dev", "test"]:
        csv_file = os.path.join(dir_path, "{}.tsv".format(fold))
        df = read_tsv_glue(csv_file, 1, True)
        if fold == "test":
            df_dict[fold] = df[["question", "sentence"]]
        else:
            df_dict[fold] = df[["question", "sentence", "label"]]
    return df_dict, None


def read_rte(dir_path):
    df_dict = dict()
    for fold in ["train", "dev", "test"]:
        csv_file = os.path.join(dir_path, "{}.tsv".format(fold))
        df = read_tsv_glue(csv_file, keep_column_names=True)
        if fold == "test":
            df_dict[fold] = df[["sentence1", "sentence2"]]
        else:
            df_dict[fold] = df[["sentence1", "sentence2", "label"]]
            assert df_dict[fold]["label"].isnull().sum().sum() == 0
    return df_dict, None


def read_wnli(dir_path):
    df_dict = dict()
    for fold in ["train", "dev", "test"]:
        csv_file = os.path.join(dir_path, "{}.tsv".format(fold))
        df = pd.read_csv(csv_file, "\t")
        if fold == "test":
            df = df[["sentence1", "sentence2"]]
        else:
            df = df[["sentence1", "sentence2", "label"]]
        df_dict[fold] = df
    return df_dict, None


# The glue diagnostic will be in MNLI
def read_glue_diagnostic(dir_path):
    csv_file = os.path.join(dir_path, "diagnostic-full.tsv")
    df = pd.read_csv(csv_file, "\t")
    df.columns = ["semantics", "predicate", "logic", "knowledge", "domain", "premise", "hypothesis", "label"]
    return df


def read_cb(dir_path):
    df_dict = dict()
    for fold in ["train", "val", "test"]:
        columns = ["premise", "hypothesis"]
        if fold != "test":
            columns.append("label")
        jsonl_path = os.path.join(dir_path, "{}.jsonl".format(fold))
        df = read_jsonl_superglue(jsonl_path)
        df = df[columns]
        df_dict[fold] = df
    return df_dict, None


def read_copa(dir_path):
    df_dict = dict()
    for fold in ["train", "val", "test"]:
        columns = ["premise", "choice1", "choice2", "question"]
        if fold != "test":
            columns.append("label")
        jsonl_path = os.path.join(dir_path, "{}.jsonl".format(fold))
        df = read_jsonl_superglue(jsonl_path)
        df = df[columns]
        df_dict[fold] = df
    return df_dict, None


# passage, question, answer, passage_idx, question_idx, answer_idx
def read_multirc(dir_path):
    df_dict = dict()
    for fold in ["train", "val", "test"]:
        columns = ["passage", "question", "answer", "psg_idx", "qst_idx", "ans_idx"]
        if fold != "test":
            columns.append("label")
        out = []
        jsonl_path = os.path.join(dir_path, "{}.jsonl".format(fold))
        with open(jsonl_path, "r") as f:
            for line in f:
                sample = json.loads(line.strip())
                psg_idx = sample["idx"]
                sample = json.loads(line.strip())
                passage = sample["passage"]["text"]
                for qa in sample["passage"]["questions"]:
                    qst_idx = qa["idx"]
                    question = qa["question"]
                    for ans in qa["answers"]:
                        ans_idx = ans["idx"]
                        answer = ans["text"]
                        if fold == "test":
                            out.append((passage, question, answer, psg_idx, qst_idx, ans_idx))
                        else:
                            label = ans["label"]
                            out.append((passage, question, answer, psg_idx, qst_idx, ans_idx, label))
        df = pd.DataFrame(out, columns=columns)
        df_dict[fold] = df
    return df_dict, None


def read_rte_superglue(dir_path):
    df_dict = dict()
    for fold in ["train", "val", "test"]:
        if fold == "test":
            columns = ["premise", "hypothesis"]
        else:
            columns = ["premise", "hypothesis", "label"]
        jsonl_path = os.path.join(dir_path, "{}.jsonl".format(fold))
        df = read_jsonl_superglue(jsonl_path)
        df = df[columns]
        df_dict[fold] = df
    return df_dict, None


def read_wic(dir_path):
    df_dict = dict()
    meta_data = dict()
    meta_data["entities1"] = {"type": "entity", "attrs": {"parent": "sentence1"}}
    meta_data["entities2"] = {"type": "entity", "attrs": {"parent": "sentence2"}}

    for fold in ["train", "val", "test"]:
        if fold != "test":
            columns = ["sentence1", "sentence2", "entities1", "entities2", "label"]
        else:
            columns = ["sentence1", "sentence2", "entities1", "entities2"]
        jsonl_path = os.path.join(dir_path, "{}.jsonl".format(fold))
        df = read_jsonl_superglue(jsonl_path)
        out = []
        for idx, row in df.iterrows():
            sentence1 = row["sentence1"]
            sentence2 = row["sentence2"]
            start1 = row["start1"]
            end1 = row["end1"]
            start2 = row["start2"]
            end2 = row["end2"]
            if fold == "test":
                out.append([sentence1, sentence2, {"start": start1, "end": end1}, {"start": start2, "end": end2}])
            else:
                label = row["label"]
                out.append(
                    [sentence1, sentence2, {"start": start1, "end": end1}, {"start": start2, "end": end2}, label]
                )
        df = pd.DataFrame(out, columns=columns)
        df_dict[fold] = df
    return df_dict, meta_data


def read_wsc(dir_path):
    df_dict = dict()
    tokenizer = WhitespaceTokenizer()
    meta_data = dict()
    meta_data["noun"] = {"type": "entity", "attrs": {"parent": "text"}}
    meta_data["pronoun"] = {"type": "entity", "attrs": {"parent": "text"}}
    for fold in ["train", "val", "test"]:
        jsonl_path = os.path.join(dir_path, "{}.jsonl".format(fold))
        df = read_jsonl_superglue(jsonl_path)
        samples = []
        for i in range(len(df)):
            text = df.loc[i, "text"]
            if fold != "test":
                label = df.loc[i, "label"]
            target = df.loc[i, "target"]
            span1_index = target["span1_index"]
            span2_index = target["span2_index"]
            span1_text = target["span1_text"]
            span2_text = target["span2_text"]
            # Build entity
            # list of entities
            # 'entities': {'start': 0, 'end': 100}
            tokens, offsets = tokenizer.encode_with_offsets(text, str)
            pos_start1 = offsets[span1_index][0]
            pos_end1 = pos_start1 + len(span1_text)
            pos_start2 = offsets[span2_index][0]
            pos_end2 = pos_start2 + len(span2_text)
            if fold == "test":
                samples.append(
                    {
                        "text": text,
                        "noun": {"start": pos_start1, "end": pos_end1},
                        "pronoun": {"start": pos_start2, "end": pos_end2},
                    }
                )
            else:
                samples.append(
                    {
                        "text": text,
                        "noun": {"start": pos_start1, "end": pos_end1},
                        "pronoun": {"start": pos_start2, "end": pos_end2},
                        "label": label,
                    }
                )
        df = pd.DataFrame(samples)
        df_dict[fold] = df
    return df_dict, meta_data


def read_boolq(dir_path):
    df_dict = dict()
    for fold in ["train", "val", "test"]:
        jsonl_path = os.path.join(dir_path, "{}.jsonl".format(fold))
        df = read_jsonl_superglue(jsonl_path)
        df_dict[fold] = df
    return df_dict, None


def read_record(dir_path):
    df_dict = dict()
    meta_data = dict()
    meta_data["entities"] = {"type": "entity", "attrs": {"parent": "text"}}
    meta_data["answers"] = {"type": "entity", "attrs": {"parent": "text"}}
    for fold in ["train", "val", "test"]:
        if fold != "test":
            columns = ["source", "text", "entities", "query", "answers"]
        else:
            columns = ["source", "text", "entities", "query"]
        jsonl_path = os.path.join(dir_path, "{}.jsonl".format(fold))
        df = read_jsonl_superglue(jsonl_path)
        df_dict[fold] = df
        out = []
        for i, row in df.iterrows():
            source = row["source"]
            passage = row["passage"]
            text = passage["text"]
            entities = passage["entities"]
            entities = [{"start": ele["start"], "end": ele["end"]} for ele in entities]
            for qas in row["qas"]:
                query = qas["query"]
                if fold != "test":
                    answer_entities = qas["answers"]
                    out.append((source, text, entities, query, answer_entities))
                else:
                    out.append((source, text, entities, query))
        df = pd.DataFrame(out, columns=columns)
        df_dict[fold] = df
    return df_dict, meta_data


def read_winogender_diagnostic(dir_path):
    jsonl_path = os.path.join(dir_path, "AX-g.jsonl")
    df = read_jsonl_superglue(jsonl_path)
    return df


def read_broadcoverage_diagnostic(dir_path):
    df = pyarrow.json.read_json(os.path.join(dir_path, "AX-b.jsonl")).to_pandas()
    return df


GLUE_TASK2PATH = {
    "cola": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/glue/cola.zip",  # noqa
    "sst": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/glue/sst.zip",  # noqa
    "mrpc": {
        "train": "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt",
        "dev": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc",
        "test": "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt",
    },
    "qqp": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/glue/qqp.zip",  # noqa
    "sts": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/glue/sts.zip",  # noqa
    "mnli": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/glue/mnli.zip",  # noqa
    "snli": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/glue/snli.zip",  # noqa
    "qnli": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/glue/qnli.zip",  # noqa
    "rte": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/glue/rte.zip",  # noqa
    "wnli": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/glue/wnli.zip",  # noqa
    "diagnostic": [
        "https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTNOrNPjR3z1cggo7WXFfrgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO77RSbfuz%2Bs95hRrYhTnByqu3U%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%2FKN%2BA9SoscRDYsnxHfG0IjXGwHN%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnRO2RDL9CapWxj%2BDl7syNyHhB7987hZ80B%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8JDCwibfhZxpaa4gd50QXQ%3D%3D",  # noqa
        "https://www.dropbox.com/s/ju7d95ifb072q9f/diagnostic-full.tsv?dl=1",
    ],
}

GLUE_READERS = {
    "cola": read_cola,
    "sst": read_sst,
    "mrpc": read_mrpc,
    "qqp": read_qqp,
    "sts": read_sts,
    "mnli": read_mnli,
    "snli": read_snli,
    "qnli": read_qnli,
    "rte": read_rte,
    "wnli": read_wnli,
    "diagnostic": read_glue_diagnostic,
}


SUPERGLUE_TASK2PATH = {
    "cb": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/superglue/cb.zip",
    "copa": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/superglue/copa.zip",
    "multirc": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/superglue/multirc.zip",
    "rte": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/superglue/rte.zip",
    "wic": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/superglue/wic.zip",
    "wsc": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/superglue/wsc.zip",
    "broadcoverage-diagnostic": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-b.zip",
    "winogender-diagnostic": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-g.zip",
    "boolq": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/superglue/boolq.zip",
    "record": "https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/datasets/text_classification/glue_superglue/superglue/record.zip",
}

SUPERGLUE_READER = {
    "cb": read_cb,
    "copa": read_copa,
    "multirc": read_multirc,
    "rte": read_rte_superglue,
    "wic": read_wic,
    "wsc": read_wsc,
    "boolq": read_boolq,
    "record": read_record,
    "broadcoverage-diagnostic": read_broadcoverage_diagnostic,
    "winogender-diagnostic": read_winogender_diagnostic,
}


def format_mrpc(data_dir):
    mrpc_dir = os.path.join(data_dir, "mrpc")
    os.makedirs(mrpc_dir, exist_ok=True)
    mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
    mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
    download(
        GLUE_TASK2PATH["mrpc"]["train"], mrpc_train_file, sha1_hash=_URL_FILE_STATS[GLUE_TASK2PATH["mrpc"]["train"]]
    )
    download(GLUE_TASK2PATH["mrpc"]["test"], mrpc_test_file, sha1_hash=_URL_FILE_STATS[GLUE_TASK2PATH["mrpc"]["test"]])
    assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
    assert os.path.isfile(mrpc_test_file), "Test data not found at %s" % mrpc_test_file
    download(
        GLUE_TASK2PATH["mrpc"]["dev"],
        os.path.join(mrpc_dir, "dev_ids.tsv"),
        sha1_hash=_URL_FILE_STATS[GLUE_TASK2PATH["mrpc"]["dev"]],
    )

    dev_ids = []
    with open(os.path.join(mrpc_dir, "dev_ids.tsv"), encoding="utf8") as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split("\t"))

    with (
        open(mrpc_train_file, encoding="utf8") as data_fh,
        open(os.path.join(mrpc_dir, "train.tsv"), "w", encoding="utf8") as train_fh,
        open(os.path.join(mrpc_dir, "dev.tsv"), "w", encoding="utf8") as dev_fh,
    ):
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split("\t")
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

    with (
        open(mrpc_test_file, encoding="utf8") as data_fh,
        open(os.path.join(mrpc_dir, "test.tsv"), "w", encoding="utf8") as test_fh,
    ):
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split("\t")
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))


def get_tasks(benchmark, task_names):
    task_names = task_names.split(",")
    ALL_TASKS = GLUE_TASKS if benchmark == "glue" else SUPERGLUE_TASKS
    if "all" in task_names:
        tasks = ALL_TASKS
    else:
        tasks = []
        for task_name in task_names:
            if task_name != "diagnostic":
                assert task_name in ALL_TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
        if "RTE" in tasks and "diagnostic" not in tasks:
            tasks.append("diagnostic")
    has_diagnostic = any(["diagnostic" in task for task in tasks])
    if has_diagnostic:
        tasks = [ele for ele in tasks if "diagnostic" not in ele]
        tasks.append("diagnostic")
    return tasks


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["glue", "superglue"], default="glue", type=str)
    parser.add_argument("-d", "--data_dir", help="directory to save data to", type=str, default=None)
    parser.add_argument(
        "-t", "--tasks", help="tasks to download data for as a comma separated string", type=str, default="all"
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=os.path.join(get_data_home_dir(), "glue"),
        help="The temporary path to download the dataset.",
    )
    return parser


def main(args):
    if args.data_dir is None:
        args.data_dir = args.benchmark
    args.cache_path = os.path.join(args.cache_path, args.benchmark)
    print('Downloading {} to "{}". Selected tasks = {}'.format(args.benchmark, args.data_dir, args.tasks))
    os.makedirs(args.cache_path, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    tasks = get_tasks(args.benchmark, args.tasks)
    if args.benchmark == "glue":
        TASK2PATH = GLUE_TASK2PATH
        TASK2READER = GLUE_READERS
    elif args.benchmark == "superglue":
        TASK2PATH = SUPERGLUE_TASK2PATH
        TASK2READER = SUPERGLUE_READER
    else:
        raise NotImplementedError
    for task in tasks:
        print("Processing {}...".format(task))
        if task == "diagnostic" or "diagnostic" in task:
            if args.benchmark == "glue":
                reader = TASK2READER[task]
                base_dir = os.path.join(args.data_dir, "rte_diagnostic")
                os.makedirs(base_dir, exist_ok=True)
                download(
                    TASK2PATH["diagnostic"][0],
                    path=os.path.join(base_dir, "diagnostic.tsv"),
                    sha1_hash=_URL_FILE_STATS[TASK2PATH["diagnostic"][0]],
                )
                download(
                    TASK2PATH["diagnostic"][1],
                    path=os.path.join(base_dir, "diagnostic-full.tsv"),
                    sha1_hash=_URL_FILE_STATS[TASK2PATH["diagnostic"][1]],
                )
                df = reader(base_dir)
                df.to_parquet(os.path.join(base_dir, "diagnostic-full.parquet"))
            else:
                for key, name in [("broadcoverage-diagnostic", "AX-b"), ("winogender-diagnostic", "AX-g")]:
                    data_file = os.path.join(args.cache_path, "{}.zip".format(key))
                    url = TASK2PATH[key]
                    reader = TASK2READER[key]
                    download(url, data_file, sha1_hash=_URL_FILE_STATS[url])
                    with zipfile.ZipFile(data_file) as zipdata:
                        zipdata.extractall(args.data_dir)
                    df = reader(os.path.join(args.data_dir, name))
                    df.to_parquet(os.path.join(args.data_dir, name, "{}.parquet".format(name)))
        elif task == "mrpc":
            reader = TASK2READER[task]
            format_mrpc(args.data_dir)
            df_dict, meta_data = reader(os.path.join(args.data_dir, "mrpc"))
            for key, df in df_dict.items():
                if key == "val":
                    key = "dev"
                df.to_parquet(os.path.join(args.data_dir, "mrpc", "{}.parquet".format(key)))
            with open(os.path.join(args.data_dir, "mrpc", "metadata.json"), "w") as f:
                json.dump(meta_data, f)
        else:
            # Download data
            data_file = os.path.join(args.cache_path, "{}.zip".format(task))
            url = TASK2PATH[task]
            reader = TASK2READER[task]
            download(url, data_file, sha1_hash=_URL_FILE_STATS[url])
            base_dir = os.path.join(args.data_dir, task)
            if os.path.exists(base_dir):
                print("Found!")
                continue
            zip_dir_name = None
            with zipfile.ZipFile(data_file) as zipdata:
                if zip_dir_name is None:
                    zip_dir_name = os.path.dirname(zipdata.infolist()[0].filename)
                zipdata.extractall(args.data_dir)
            shutil.move(os.path.join(args.data_dir, zip_dir_name), base_dir)
            df_dict, meta_data = reader(base_dir)
            for key, df in df_dict.items():
                if key == "val":
                    key = "dev"
                df.to_parquet(os.path.join(base_dir, "{}.parquet".format(key)))
            if meta_data is not None:
                with open(os.path.join(base_dir, "metadata.json"), "w") as f:
                    json.dump(meta_data, f)
        print("\tCompleted!")


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
