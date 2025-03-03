import json
import os

import pandas as pd

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import download

from ..utils import get_home_dir


def download_ecommerce():
    download(
        "https://raw.githubusercontent.com/allanj/ner_incomplete_annotation/master/data/ecommerce/train.txt",
        os.path.join(get_home_dir(), "train.txt"),
    )
    download(
        "https://raw.githubusercontent.com/allanj/ner_incomplete_annotation/master/data/ecommerce/dev.txt",
        os.path.join(get_home_dir(), "dev.txt"),
    )


def read_bio(sample):
    raw_string = ""
    entities = []
    new_entity = None
    prev_bio_type = None
    ptr = 0

    # Parse the data from the BIO format
    for ele in sample.split("\n"):
        dat = ele.split()
        if len(dat) != 2:
            continue
        token, bio_label = dat
        raw_string += token
        new_ptr = ptr + len(token)
        if bio_label.startswith("O"):
            bio_type, label = "O", None
        else:
            bio_type, label = bio_label.split("-")
        if bio_type == "O":
            if new_entity:
                entities.append(new_entity)
            new_entity = None
        elif bio_type == "B":
            if new_entity:
                entities.append(new_entity)
            new_entity = {"entity_group": label, "start": ptr, "end": new_ptr}
        elif bio_type == "I":
            if prev_bio_type in ["B", "I"]:
                # Keep update the new_entity
                assert new_entity["entity_group"] == label
                new_entity["end"] = new_ptr
            else:
                new_entity = {"entity_group": label, "start": ptr, "end": new_ptr}
        else:
            raise NotImplementedError
        ptr = new_ptr
        prev_bio_type = bio_type
    if new_entity:
        entities.append(new_entity)
    return raw_string, entities


def bio_samples_to_df(samples):
    raw_strings = []
    entity_list = []
    for sample in samples:
        raw_string, entities = read_bio(sample)
        raw_strings.append(raw_string)
        entity_list.append(json.dumps(entities))
    df = pd.DataFrame({"text_snippet": raw_strings, "entity_annotations": entity_list})
    return df


def get_data():
    train_data = open(os.path.join(get_home_dir(), "dev.txt"), encoding="utf-8").read()
    train_df = bio_samples_to_df(train_data.split("\n\n"))

    dev_data = open(os.path.join(get_home_dir(), "dev.txt"), encoding="utf-8").read()
    dev_df = bio_samples_to_df(dev_data.split("\n\n"))

    return train_df, dev_df


def test_ner_chinese():
    download_ecommerce()
    train_df, dev_df = get_data()
    predictor = MultiModalPredictor(
        problem_type="ner",
        label="entity_annotations",
    )
    predictor.fit(
        train_data=train_df,
        tuning_data=dev_df,
        hyperparameters={
            "model.ner_text.checkpoint_name": "hfl/chinese-lert-small",
            "optim.top_k": 1,
            "env.num_gpus": -1,
            "optim.max_epochs": 1,
        },
    )
    predictor.evaluate(dev_df)
    predictor.predict(dev_df.head(2))
