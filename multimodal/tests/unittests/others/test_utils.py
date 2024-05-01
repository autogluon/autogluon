import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from ray import tune
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from transformers import AutoTokenizer

from autogluon.multimodal.constants import (
    BINARY,
    CATEGORICAL,
    CLASSIFICATION,
    DATA,
    ENVIRONMENT,
    IMAGE_PATH,
    MODEL,
    MULTICLASS,
    NER,
    NER_ANNOTATION,
    NUMERICAL,
    OBJECT_DETECTION,
    OPTIMIZATION,
    REGRESSION,
    TEXT,
    TEXT_NER,
)
from autogluon.multimodal.data.infer_types import infer_ner_column_type, infer_problem_type
from autogluon.multimodal.data.label_encoder import CustomLabelEncoder
from autogluon.multimodal.data.utils import process_ner_annotations
from autogluon.multimodal.utils import (
    apply_omegaconf_overrides,
    data_to_df,
    filter_hyperparameters,
    filter_search_space,
    get_config,
    get_default_config,
    is_url,
    merge_bio_format,
    parse_dotlist_conf,
    split_hyperparameters,
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
    "labels,positive_class",
    [
        ([1, 2, 2], 2),
        ([1, 2, 2], 1),
        ([1, 0, 1, 0, 0], 0),
        ([1, 0, 1, 0, 0], None),
        (["a", "e", "e", "a"], "a"),
        (["b", "d", "b", "d"], "d"),
        (["a", "d", "e", "b"], "d"),
        ([3, 2, 1, 0], 2),
    ],
)
def test_label_encoder(labels, positive_class):
    label_encoder = CustomLabelEncoder(positive_class=positive_class)
    label_encoder.fit(labels)

    # test encoding positive class
    if positive_class:
        assert label_encoder.transform([positive_class]).item() == len(label_encoder.classes_) - 1
    else:
        assert label_encoder.transform([label_encoder.classes_[-1]]).item() == len(label_encoder.classes_) - 1

    # test encoding
    sklearn_le = LabelEncoder()
    sklearn_le.fit(labels)
    sk_encoded_labels = sklearn_le.transform(labels)
    our_encoded_labels = label_encoder.transform(labels)
    if positive_class:
        sk_pos_label = sklearn_le.transform([positive_class]).item()
    else:
        sk_pos_label = len(sklearn_le.classes_) - 1
    sk_encoded_labels[sk_encoded_labels == sk_pos_label] = len(sklearn_le.classes_)
    sk_encoded_labels[sk_encoded_labels > sk_pos_label] -= 1
    sk_encoded_labels[sk_encoded_labels < 0] = 0
    assert (sk_encoded_labels == our_encoded_labels).all()

    # test inverse encoding
    assert label_encoder.inverse_transform(our_encoded_labels).tolist() == labels


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
    assert res == [3, 4, 1, 1, 1, 1, 1, 5, 6, 1, 1, 1, 7, 8, 8, 8], "Labelling is wrong!"


@pytest.mark.parametrize(
    "hyperparameters,column_types,model_in_config,fit_called,result",
    [
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"a": IMAGE_PATH, "b": TEXT, "c": NUMERICAL, "d": CATEGORICAL},
            None,
            False,
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
        ),
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"a": IMAGE_PATH},
            None,
            False,
            {
                "model.names": ["timm_image", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
            },
        ),
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"b": TEXT},
            None,
            False,
            {
                "model.names": ["hf_text", "fusion_mlp"],
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
        ),
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"b": TEXT, "c": NUMERICAL},
            None,
            False,
            {
                "model.names": ["numerical_mlp", "hf_text", "fusion_mlp"],
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
        ),
        (
            {
                "model.names": ["timm_image"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"a": IMAGE_PATH},
            None,
            False,
            {
                "model.names": ["timm_image"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
            },
        ),
        (
            {
                "model.names": ["hf_text"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"b": TEXT, "c": NUMERICAL, "d": CATEGORICAL},
            None,
            False,
            {
                "model.names": ["hf_text"],
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
        ),
        (
            {
                "model.names": ["hf_text"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"c": NUMERICAL, "d": CATEGORICAL},
            None,
            False,
            AssertionError,
        ),
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"a": IMAGE_PATH, "b": TEXT, "c": NUMERICAL, "d": CATEGORICAL},
            "timm_image",
            False,
            {
                "model.names": ["timm_image"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
            },
        ),
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"a": IMAGE_PATH, "b": TEXT, "c": NUMERICAL, "d": CATEGORICAL},
            "numerical_mlp",
            False,
            {
                "model.names": ["numerical_mlp"],
            },
        ),
        (
            {
                "model.names": ["categorical_mlp", "numerical_mlp", "fusion_mlp"],
            },
            {"a": IMAGE_PATH, "b": TEXT, "c": NUMERICAL, "d": CATEGORICAL},
            "hf_text",
            False,
            AssertionError,
        ),
        (
            {
                "model.names": ["timm_image", "hf_text", "fusion_mlp"],
                "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
                "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            },
            {"a": IMAGE_PATH, "b": TEXT},
            None,
            True,
            {},
        ),
    ],
)
def test_filter_hyperparameters(hyperparameters, column_types, model_in_config, fit_called, result):
    if model_in_config:
        config = get_default_config()
        config.model.names = [model_in_config]
        model_keys = list(config.model.keys())
        for key in model_keys:
            if key != "names" and key != model_in_config:
                delattr(config.model, key)
    else:
        config = None

    if result == AssertionError:
        with pytest.raises(AssertionError):
            filtered_hyperparameters = filter_hyperparameters(
                hyperparameters=hyperparameters,
                column_types=column_types,
                config=config,
                fit_called=fit_called,
            )
    else:
        filtered_hyperparameters = filter_hyperparameters(
            hyperparameters=hyperparameters,
            column_types=column_types,
            config=config,
            fit_called=fit_called,
        )

        assert filtered_hyperparameters == result


@pytest.mark.parametrize(
    "train_transforms,val_transforms,empty_advanced_hyperparameters",
    [
        (
            ["resize_shorter_side", "center_crop", "random_horizontal_flip", "color_jitter"],
            ["resize_shorter_side", "center_crop"],
            True,
        ),
        (
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()],
            [transforms.Resize(256), transforms.CenterCrop(224)],
            False,
        ),
    ],
)
def test_split_hyperparameters(train_transforms, val_transforms, empty_advanced_hyperparameters):
    hyperparameters = {
        "model.timm_image.train_transforms": train_transforms,
        "model.timm_image.val_transforms": val_transforms,
    }
    hyperparameters, advanced_hyperparameters = split_hyperparameters(hyperparameters)
    if empty_advanced_hyperparameters:
        assert not advanced_hyperparameters
    else:
        assert advanced_hyperparameters


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


@pytest.mark.parametrize(
    "y_data,provided_problem_type,gt_problem_type",
    [
        (pd.Series([0, 1, 0, 1, 1, 0]), None, BINARY),
        (pd.Series(["a", "b", "c"]), None, MULTICLASS),
        (pd.Series(["a", "b", "c"]), CLASSIFICATION, MULTICLASS),
        (pd.Series(np.linspace(0.0, 1.0, 100)), None, REGRESSION),
        (pd.Series(["0", "1", "2", 3, 4, 5, 5, 5, 0]), None, MULTICLASS),
        (None, NER, NER),
        (None, OBJECT_DETECTION, OBJECT_DETECTION),
    ],
)
def test_infer_problem_type(y_data, provided_problem_type, gt_problem_type):
    inferred_problem_type = infer_problem_type(
        y_train_data=y_data,
        provided_problem_type=provided_problem_type,
    )
    assert inferred_problem_type == gt_problem_type
