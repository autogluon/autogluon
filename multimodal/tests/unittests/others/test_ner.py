import json
import os
import shutil
import tempfile
from collections import OrderedDict
from unittest import mock

import numpy.testing as npt
import pandas as pd
import pytest
import torch
from ray import tune
from transformers import AutoTokenizer

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import (
    IMAGE_PATH,
    NER,
    NER_ANNOTATION,
    TEXT,
    TEXT_NER,
)
from autogluon.multimodal.data import NerProcessor, infer_ner_column_type
from autogluon.multimodal.utils import merge_bio_format, visualize_ner

from ..utils import get_home_dir


def get_data():
    sample = {
        "text_snippet": [
            "EU rejects German call to boycott British lamb .",
            "Peter Blackburn",
            "He said further scientific study was required and if it was found that action was needed it should be taken by the European Union .",
            ".",
            "It brought in 4,275 tonnes of British mutton , some 10 percent of overall imports .",
        ],
        "entity_annotations": [
            '[{"entity_group": "B-ORG", "start": 0, "end": 2}, {"entity_group": "B-MISC", "start": 11, "end": 17}, {"entity_group": "B-MISC", "start": 34, "end": 41}]',
            '[{"entity_group": "B-PER", "start": 0, "end": 5}, {"entity_group": "I-PER", "start": 6, "end": 15}]',
            '[{"entity_group": "B-ORG", "start": 115, "end": 123}, {"entity_group": "I-ORG", "start": 124, "end": 129}]',
            "[]",
            '[{"entity_group": "B-MISC", "start": 30, "end": 37}]',
        ],
    }
    return pd.DataFrame.from_dict(sample)


@pytest.mark.parametrize(
    "checkpoint_name,searcher,scheduler",
    [
        ("google/electra-small-discriminator", None, None),
        ("google/electra-small-discriminator", "bayes", "FIFO"),
    ],
)
def test_ner(checkpoint_name, searcher, scheduler):
    train_data = get_data()
    label_col = "entity_annotations"

    if searcher is None:
        lr = 0.0001
        hyperparameter_tune_kwargs = None
    else:
        lr = tune.uniform(0.0001, 0.01)
        hyperparameter_tune_kwargs = {"num_trials": 2, "searcher": searcher, "scheduler": scheduler}
    predictor = MultiModalPredictor(problem_type="ner", label=label_col)
    predictor.fit(
        train_data=train_data,
        time_limit=60,
        hyperparameters={"model.ner_text.checkpoint_name": checkpoint_name, "optim.lr": lr},
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )

    scores = predictor.evaluate(train_data)

    test_predict = train_data.drop(label_col, axis=1)
    test_predict = test_predict.head(2)
    predictions = predictor.predict(test_predict)
    embeddings = predictor.extract_embedding(test_predict)


@pytest.mark.parametrize(
    "checkpoint_name",
    [("google/electra-small-discriminator")],
)
def test_multi_column_ner(checkpoint_name):
    train_data = get_data()
    train_data["add_text"] = train_data.text_snippet
    label_col = "entity_annotations"
    predictor = MultiModalPredictor(problem_type="ner", label=label_col)
    predictor.fit(
        train_data=train_data,
        time_limit=10,
        column_types={"text_snippet": "text_ner"},
        hyperparameters={"model.ner_text.checkpoint_name": checkpoint_name},
    )
    scores = predictor.evaluate(train_data)
    test_predict = train_data.drop(label_col, axis=1)
    test_predict = test_predict.head(2)
    predictions = predictor.predict(test_predict)
    proba = predictor.predict_proba(test_predict)
    embeddings = predictor.extract_embedding(test_predict)


def test_ner_standalone():
    requests_gag = mock.patch(
        "requests.Session.request",
        mock.Mock(side_effect=RuntimeError("Please use the `responses` library to mock HTTP in your tests.")),
    )
    train_data = get_data()
    label_col = "entity_annotations"
    test_data = train_data.drop(label_col, axis=1)

    predictor = MultiModalPredictor(problem_type="ner", label=label_col)

    save_path = os.path.join(get_home_dir(), "standalone", "false")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    predictor.fit(
        train_data=train_data,
        time_limit=40,
        hyperparameters={"model.ner_text.checkpoint_name": "google/electra-small-discriminator"},
        save_path=save_path,
    )
    save_path_standalone = os.path.join(get_home_dir(), "standalone", "true")

    predictor.save(
        path=save_path_standalone,
        standalone=True,
    )

    del predictor
    torch.cuda.empty_cache()

    loaded_online_predictor = MultiModalPredictor.load(path=save_path)
    online_predictions = loaded_online_predictor.predict(test_data, as_pandas=False)
    del loaded_online_predictor

    with requests_gag:
        # No internet connection here. If any command require internet connection, a RuntimeError will be raised!
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.hub.set_dir(tmpdirname)  # block reading files in `.cache`.
            loaded_offline_predictor = MultiModalPredictor.load(path=save_path_standalone)

    offline_predictions = loaded_offline_predictor.predict(test_data, as_pandas=False)
    del loaded_offline_predictor
    npt.assert_equal(online_predictions[0], offline_predictions[0])


def test_merge_bio():
    sentence = "Game of Thrones is an American fantasy drama television series created by David Benioff"
    predictions = [
        [
            {"entity_group": "B-TITLE", "start": 0, "end": 4},
            {"entity_group": "I-TITLE", "start": 5, "end": 7},
            {"entity_group": "I-TITLE", "start": 8, "end": 15},
            {"entity_group": "B-GENRE", "start": 22, "end": 30},
            {"entity_group": "B-GENRE", "start": 31, "end": 38},
            {"entity_group": "I-GENRE", "start": 39, "end": 44},
            {"entity_group": "B-DIRECTOR", "start": 74, "end": 79},
            {"entity_group": "I-DIRECTOR", "start": 80, "end": 87},
        ]
    ]
    res = merge_bio_format([sentence], predictions)
    expected_res = [
        [
            {"entity_group": "TITLE", "start": 0, "end": 15},
            {"entity_group": "GENRE", "start": 22, "end": 30},
            {"entity_group": "GENRE", "start": 31, "end": 44},
            {"entity_group": "DIRECTOR", "start": 74, "end": 87},
        ]
    ]
    assert res == expected_res, f"Wrong results {res} from merge_bio_format!"


def test_misc_visualize_ner():
    sentence = "Albert Einstein was born in Germany and is widely acknowledged to be one of the greatest physicists."
    annotation = [
        {"entity_group": "PERSON", "start": 0, "end": 15},
        {"entity_group": "LOCATION", "start": 28, "end": 35},
    ]
    visualize_ner(sentence, annotation)

    # Test using string for annotation
    visualize_ner(sentence, json.dumps(annotation))


def test_process_ner_annotations():
    text = "SwissGear Sion Softside Expandable Roller Luggage, Dark Grey, Checked-Medium 25-Inch"
    annotation = [((0, 14), "Brand"), ((50, 60), "Color"), ((70, 85), "Dimensions")]
    entity_map = {
        "X": 1,
        "O": 2,
        "B-Brand": 3,
        "I-Brand": 4,
        "B-Color": 5,
        "I-Color": 6,
        "B-Dimensions": 7,
        "I-Dimensions": 8,
    }
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    tokenizer.model_max_length = 512
    res = NerProcessor.process_ner_annotations(annotation, text, entity_map, tokenizer, is_eval=True)[0]
    assert res == [3, 4, 1, 1, 1, 1, 1, 5, 6, 1, 1, 1, 7, 8, 8, 8], "Labelling is wrong!"


@pytest.mark.parametrize(
    "column_types,gt_column_types",
    [
        (
            {
                "abc": TEXT,
                "label": NER_ANNOTATION,
            },
            {
                "abc": TEXT_NER,
                "label": NER_ANNOTATION,
            },
        ),
        (
            {
                "abc": TEXT_NER,
                "label": NER_ANNOTATION,
            },
            {
                "abc": TEXT_NER,
                "label": NER_ANNOTATION,
            },
        ),
        (
            {
                "abc": TEXT,
                "xyz": TEXT,
                "label": NER_ANNOTATION,
            },
            {
                "abc": TEXT_NER,
                "xyz": TEXT,
                "label": NER_ANNOTATION,
            },
        ),
        (
            {
                "abc": TEXT,
                "xyz": TEXT,
                "efg": IMAGE_PATH,
                "label": NER_ANNOTATION,
            },
            {
                "abc": TEXT_NER,
                "xyz": TEXT,
                "efg": IMAGE_PATH,
                "label": NER_ANNOTATION,
            },
        ),
    ],
)
def test_infer_ner_column_type(column_types, gt_column_types):
    column_types = OrderedDict(column_types)
    gt_column_types = OrderedDict(gt_column_types)
    column_types = infer_ner_column_type(column_types)
    assert column_types == gt_column_types
