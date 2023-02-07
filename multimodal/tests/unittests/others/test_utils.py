import json

import pandas as pd
import pytest
from omegaconf import OmegaConf
from ray import tune
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from autogluon.multimodal.constants import BINARY, DATA, ENVIRONMENT, MODEL, MULTICLASS, OPTIMIZATION
from autogluon.multimodal.data.preprocess_dataframe import MultiModalFeaturePreprocessor
from autogluon.multimodal.data.utils import process_ner_annotations
from autogluon.multimodal.utils import (
    apply_omegaconf_overrides,
    data_to_df,
    filter_search_space,
    get_config,
    is_url,
    merge_bio_format,
    parse_dotlist_conf,
    try_to_infer_pos_label,
    visualize_ner,
)


@pytest.mark.parametrize(
    "hyperparameters, keys_to_filter, expected",
    [
        ({"model.abc": tune.choice(["a", "b"])}, ["model"], {}),
        ({"model.abc": tune.choice(["a", "b"])}, ["data"], {"model.abc": tune.choice(["a", "b"])}),
        ({"model.abc": "def"}, ["model"], {"model.abc": "def"}),
        (
            {
                "data.abc.def": tune.choice(["a", "b"]),
                "model.abc": "def",
                "environment.abc.def": tune.choice(["a", "b"]),
            },
            ["data"],
            {"model.abc": "def", "environment.abc.def": tune.choice(["a", "b"])},
        ),
    ],
)
def test_filter_search_space(hyperparameters, keys_to_filter, expected):
    # We test keys here because the object might be copied and hence direct comparison will fail
    assert filter_search_space(hyperparameters, keys_to_filter).keys() == expected.keys()


@pytest.mark.parametrize("hyperparameters, keys_to_filter", [({"model.abc": tune.choice(["a", "b"])}, ["abc"])])
def test_invalid_filter_search_space(hyperparameters, keys_to_filter):
    with pytest.raises(Exception) as e_info:
        filter_search_space(hyperparameters, keys_to_filter)


@pytest.mark.parametrize(
    "data,expected",
    [
        ("aaa=a bbb=b ccc=c", {"aaa": "a", "bbb": "b", "ccc": "c"}),
        ("a.a.aa=b b.b.bb=c", {"a.a.aa": "b", "b.b.bb": "c"}),
        ("a.a.aa=1 b.b.bb=100", {"a.a.aa": "1", "b.b.bb": "100"}),
        (["a.a.aa=1", "b.b.bb=100"], {"a.a.aa": "1", "b.b.bb": "100"}),
    ],
)
def test_parse_dotlist_conf(data, expected):
    assert parse_dotlist_conf(data) == expected


def test_apply_omegaconf_overrides():
    conf = OmegaConf.from_dotlist(["a.aa.aaa=[1, 2, 3, 4]", "a.aa.bbb=2", "a.bb.aaa='100'", "a.bb.bbb=4"])
    overrides = "a.aa.aaa=[1, 3, 5] a.aa.bbb=3"
    new_conf = apply_omegaconf_overrides(conf, overrides.split())
    assert new_conf.a.aa.aaa == [1, 3, 5]
    assert new_conf.a.aa.bbb == 3
    new_conf2 = apply_omegaconf_overrides(conf, {"a.aa.aaa": [1, 3, 5, 7], "a.aa.bbb": 4})
    assert new_conf2.a.aa.aaa == [1, 3, 5, 7]
    assert new_conf2.a.aa.bbb == 4

    with pytest.raises(KeyError):
        new_conf3 = apply_omegaconf_overrides(conf, {"a.aa.aaaaaa": [1, 3, 5, 7], "a.aa.bbb": 4})


@pytest.mark.parametrize(
    "labels,pos_label,problem_type,true_pos_label",
    [
        ([1, 2, 2], 2, BINARY, 1),
        ([1, 2, 2], 1, BINARY, 0),
        ([1, 0, 1, 0, 0], 0, BINARY, 0),
        ([1, 0, 1, 0, 0], None, BINARY, 1),
        (["a", "e", "e", "a"], "a", BINARY, 0),
        (["b", "d", "b", "d"], "d", BINARY, 1),
        (["a", "d", "e", "b"], "d", MULTICLASS, None),
        ([3, 2, 1, 0], 2, MULTICLASS, None),
    ],
)
def test_inferring_pos_label(labels, pos_label, problem_type, true_pos_label):
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    overrides = {}
    if pos_label is not None:
        overrides.update(
            {
                "data.pos_label": pos_label,
            }
        )
    config = get_config(
        config=config,
        overrides=overrides,
    )
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    pos_label = try_to_infer_pos_label(
        data_config=config.data,
        label_encoder=label_encoder,
        problem_type=problem_type,
    )
    assert pos_label == true_pos_label


@pytest.mark.parametrize(
    "data,required_columns,all_columns,is_valid_input",
    [
        ([1, 2, 3], ["a"], ["a"], True),
        ([1, 2, 3], ["a"], ["a", "b"], False),
        ({"a": [1, 2, 3]}, ["a"], ["a", "b"], True),
        ({"a": [1, 2, 3]}, ["b"], ["a", "b"], False),
        ({"a": [1, 2, 3], "b": [4, 5, 6]}, ["b", "a"], ["a", "b"], True),
        ([(1, 2), (3, 4)], ["a", "b"], ["a", "b"], True),
        ([[1, 2], [3, 4]], ["a"], ["a", "b"], True),
        ([[1, 2], [3, 4]], ["a", "b"], ["a", "b", "c"], False),
        ([{"a": 1, "b": 2, "c": 3}, {"a": 10, "b": 20, "c": 30}], ["a", "b", "c"], ["a", "b", "c"], True),
        ([{"a": 1, "b": 2, "c": 3}, {"a": 10, "b": 20, "c": 30}], ["a", "c"], ["a", "b", "c"], True),
        ([{"a": 1, "b": 2, "c": 3}, {"a": 10, "b": 20, "c": 30}], ["a", "b", "c", "d"], ["a", "b", "c", "d"], False),
        ([{"a": 1, "b": 2, "c": 3}, {"a": 10, "b": 20, "c": 30}], ["a", "b", "d"], ["a", "b", "c", "d"], False),
        (pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), ["b", "a"], ["b", "c", "a"], True),
    ],
)
def test_data_to_df(data, required_columns, all_columns, is_valid_input):
    if is_valid_input:
        df = data_to_df(data=data, required_columns=required_columns, all_columns=all_columns)
    else:
        with pytest.raises(ValueError):
            df = data_to_df(data=data, required_columns=required_columns, all_columns=all_columns)


@pytest.mark.parametrize(
    "path,is_valid_url",
    [
        ("/media/data/coco17/annotations/instances_val2017.json", False),
        ("This is a test.", False),
        ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", True),
        ("http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar", True),
        (
            "https://automl-mm-bench.s3.amazonaws.com/voc_script/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth",
            True,
        ),
    ],
)
def test_is_url(path, is_valid_url):
    assert is_url(path) == is_valid_url


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
    res = process_ner_annotations(annotation, text, entity_map, tokenizer, is_eval=True)[0]
    assert res == [3, 4, 1, 1, 1, 1, 5, 6, 7, 8], "Labelling is wrong!"
