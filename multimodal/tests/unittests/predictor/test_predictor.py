import copy
import os
import shutil
import tempfile
import uuid

import numpy.testing as npt
import pytest
from omegaconf import OmegaConf
from torch import nn

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import (
    BEST,
    BINARY,
    BIT_FIT,
    DATA,
    DISTILLER,
    ENVIRONMENT,
    GREEDY_SOUP,
    IA3,
    LORA,
    LORA_BIAS,
    LORA_NORM,
    MODEL,
    MULTICLASS,
    NORM_FIT,
    OPTIMIZATION,
    UNIFORM_SOUP,
)
from autogluon.multimodal.utils import modify_duplicate_model_names
from autogluon.multimodal.utils.misc import shopee_dataset

from ..utils.unittest_datasets import AEDataset, HatefulMeMesDataset, PetFinderDataset
from ..utils.utils import get_home_dir

ALL_DATASETS = {
    "petfinder": PetFinderDataset(),
    "petfinder_bytearray": PetFinderDataset(is_bytearray=True),
    "hateful_memes": HatefulMeMesDataset(),
    "hateful_memes_bytearray": HatefulMeMesDataset(is_bytearray=True),
    "ae": AEDataset(),
}


def verify_predictor_save_load(predictor, df, verify_embedding=True, cls=MultiModalPredictor):
    root = str(uuid.uuid4())
    os.makedirs(root, exist_ok=True)
    predictor.save(root)
    predictions = predictor.predict(df, as_pandas=False)
    # Test fit_summary()
    predictor.fit_summary()

    loaded_predictor = cls.load(root)
    # Test fit_summary()
    loaded_predictor.fit_summary()

    predictions2 = loaded_predictor.predict(df, as_pandas=False)
    predictions2_df = loaded_predictor.predict(df, as_pandas=True)
    npt.assert_equal(predictions, predictions2)
    npt.assert_equal(predictions2, predictions2_df.to_numpy())
    if predictor.problem_type in [BINARY, MULTICLASS]:
        predictions_prob = predictor.predict_proba(df, as_pandas=False)
        predictions2_prob = loaded_predictor.predict_proba(df, as_pandas=False)
        predictions2_prob_df = loaded_predictor.predict_proba(df, as_pandas=True)
        npt.assert_equal(predictions_prob, predictions2_prob)
        npt.assert_equal(predictions2_prob, predictions2_prob_df.to_numpy())
    if verify_embedding:
        embeddings = predictor.extract_embedding(df)
        assert embeddings.shape[0] == len(df)
    shutil.rmtree(root)


def verify_realtime_inference(predictor, df, verify_embedding=True):
    for i in range(1, 3):
        df_small = df.head(i)
        predictions_default = predictor.predict(df_small, as_pandas=False, realtime=False)
        predictions_realtime = predictor.predict(df_small, as_pandas=False, realtime=True)
        npt.assert_equal(predictions_default, predictions_realtime)
        if predictor.problem_type in [BINARY, MULTICLASS]:
            predictions_prob_default = predictor.predict_proba(df_small, as_pandas=False, realtime=False)
            predictions_prob_realtime = predictor.predict_proba(df_small, as_pandas=False, realtime=True)
            npt.assert_equal(predictions_prob_default, predictions_prob_realtime)
        if verify_embedding:
            embeddings_default = predictor.extract_embedding(df_small, realtime=False)
            embeddings_realtime = predictor.extract_embedding(df_small, realtime=True)
            npt.assert_equal(embeddings_default, embeddings_realtime)


def verify_no_redundant_model_configs(predictor):
    model_names = list(predictor._learner._config.model.keys())
    model_names.remove("names")
    assert sorted(predictor._learner._config.model.names) == sorted(model_names)


@pytest.mark.parametrize(
    "dataset_name,model_names,text_backbone,image_backbone,top_k_average_method,efficient_finetune,loss_function",
    [
        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "fusion_mlp"],
            "nlpaueb/legal-bert-small-uncased",
            "swin_tiny_patch4_window7_224",
            GREEDY_SOUP,
            LORA,
            "auto",
        ),
        (
            "petfinder",
            ["t_few"],
            "t5-small",
            None,
            BEST,
            IA3,
            "auto",
        ),
        (
            "hateful_memes",
            ["timm_image", "t_few", "fusion_mlp"],
            "t5-small",
            "mobilenetv3_small_100",
            BEST,
            IA3,
            "auto",
        ),
        (
            "hateful_memes_bytearray",
            ["timm_image", "hf_text", "fusion_mlp"],
            "monsoon-nlp/hindi-bert",
            "mobilenetv3_small_100",
            UNIFORM_SOUP,
            LORA_NORM,
            "auto",
        ),
        (
            "petfinder_bytearray",
            ["ft_transformer", "timm_image", "fusion_mlp"],
            None,
            "mobilenetv3_small_100",
            GREEDY_SOUP,
            None,
            "auto",
        ),
        (
            "petfinder",
            ["ft_transformer", "hf_text", "fusion_mlp"],
            "nlpaueb/legal-bert-small-uncased",
            None,
            UNIFORM_SOUP,
            None,
            "auto",
        ),
        (
            "petfinder",
            ["ft_transformer"],
            None,
            None,
            BEST,
            BIT_FIT,
            "auto",
        ),
        (
            "hateful_memes",
            ["timm_image"],
            None,
            "mobilenetv3_small_100",
            UNIFORM_SOUP,
            NORM_FIT,
            "auto",
        ),
        (
            "ae",
            ["hf_text"],
            "nlpaueb/legal-bert-small-uncased",
            None,
            BEST,
            LORA_BIAS,
            "bcewithlogitsloss",
        ),
        (
            "ae",
            ["hf_text"],
            "CLTL/MedRoBERTa.nl",
            None,
            BEST,
            None,
            "auto",
        ),
        (
            "hateful_memes",
            ["clip"],
            None,
            None,
            BEST,
            NORM_FIT,
            "auto",
        ),
    ],
)
def test_predictor_basic(
    dataset_name,
    model_names,
    text_backbone,
    image_backbone,
    top_k_average_method,
    efficient_finetune,
    loss_function,
):
    dataset = ALL_DATASETS[dataset_name]
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": model_names,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "optimization.top_k_average_method": top_k_average_method,
        "optimization.efficient_finetune": efficient_finetune,
        "optimization.loss_function": loss_function,
        "data.categorical.convert_to_text": False,  # ensure the categorical model is used.
        "data.numerical.convert_to_text": False,  # ensure the numerical model is used.
    }
    if text_backbone is not None:
        if "t_few" in model_names:
            hyperparameters.update(
                {
                    "model.t_few.checkpoint_name": "t5-small",
                    "model.t_few.gradient_checkpointing": False,
                }
            )
        else:
            hyperparameters.update(
                {
                    "model.hf_text.checkpoint_name": text_backbone,
                }
            )
    if image_backbone is not None:
        hyperparameters.update(
            {
                "model.timm_image.checkpoint_name": image_backbone,
            }
        )
    save_path = os.path.join(get_home_dir(), "outputs", dataset_name)
    if text_backbone is not None:
        save_path = os.path.join(save_path, text_backbone)
    if image_backbone is not None:
        save_path = os.path.join(save_path, image_backbone)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=20,
        save_path=save_path,
    )
    verify_no_redundant_model_configs(predictor)
    score = predictor.evaluate(dataset.test_df)
    verify_predictor_save_load(predictor, dataset.test_df)

    # Test for continuous fit
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=20,
    )
    verify_no_redundant_model_configs(predictor)
    verify_predictor_save_load(predictor, dataset.test_df)

    # Saving to folder, loading the saved model and call fit again (continuous fit)
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictor = MultiModalPredictor.load(root)
        predictor.fit(
            train_data=dataset.train_df,
            hyperparameters=hyperparameters,
            time_limit=10,
        )


@pytest.mark.single_gpu
@pytest.mark.parametrize(
    "dataset_name,model_names,text_backbone,image_backbone,top_k_average_method,efficient_finetune,loss_function",
    [
        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "fusion_mlp"],
            "nlpaueb/legal-bert-small-uncased",
            "swin_tiny_patch4_window7_224",
            GREEDY_SOUP,
            LORA,
            "auto",
        ),
        (
            "hateful_memes",
            ["timm_image", "t_few", "fusion_mlp"],
            "t5-small",
            "mobilenetv3_small_100",
            BEST,
            IA3,
            "auto",
        ),
        (
            "hateful_memes",
            ["clip"],
            None,
            None,
            BEST,
            NORM_FIT,
            "auto",
        ),
    ],
)
def test_predictor_realtime_inference(
    dataset_name,
    model_names,
    text_backbone,
    image_backbone,
    top_k_average_method,
    efficient_finetune,
    loss_function,
):
    dataset = ALL_DATASETS[dataset_name]
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": model_names,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "optimization.top_k_average_method": top_k_average_method,
        "optimization.efficient_finetune": efficient_finetune,
        "optimization.loss_function": loss_function,
        "data.categorical.convert_to_text": False,  # ensure the categorical model is used.
        "data.numerical.convert_to_text": False,  # ensure the numerical model is used.
    }
    if text_backbone is not None:
        if "t_few" in model_names:
            hyperparameters.update(
                {
                    "model.t_few.checkpoint_name": "t5-small",
                    "model.t_few.gradient_checkpointing": False,
                }
            )
        else:
            hyperparameters.update(
                {
                    "model.hf_text.checkpoint_name": text_backbone,
                }
            )
    if image_backbone is not None:
        hyperparameters.update(
            {
                "model.timm_image.checkpoint_name": image_backbone,
            }
        )
    save_path = os.path.join(get_home_dir(), "outputs", dataset_name)
    if text_backbone is not None:
        save_path = os.path.join(save_path, text_backbone)
    if image_backbone is not None:
        save_path = os.path.join(save_path, image_backbone)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=20,
        save_path=save_path,
    )
    verify_realtime_inference(predictor, dataset.test_df)


def test_predictor_standalone():  # test standalone feature in MultiModalPredictor.save()
    from unittest import mock

    import torch

    requests_gag = mock.patch(
        "requests.Session.request",
        mock.Mock(side_effect=RuntimeError("Please use the `responses` library to mock HTTP in your tests.")),
    )

    dataset = ALL_DATASETS["petfinder"]

    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "fusion_mlp", "t_few"],
        "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "model.t_few.checkpoint_name": "t5-small",
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    save_path = os.path.join(get_home_dir(), "standalone", "false")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=30,
        save_path=save_path,
    )

    save_path_standalone = os.path.join(get_home_dir(), "standalone", "true")

    predictor.save(
        path=save_path_standalone,
        standalone=True,
    )

    # make sure the dumping doesn't affect predictor loading
    predictor.dump_model()

    del predictor
    torch.cuda.empty_cache()

    loaded_online_predictor = MultiModalPredictor.load(path=save_path)
    online_predictions = loaded_online_predictor.predict(dataset.test_df, as_pandas=False)
    del loaded_online_predictor

    # Check if the predictor can be loaded from an offline environment.
    with requests_gag:
        # No internet connection here. If any command require internet connection, a RuntimeError will be raised.
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.hub.set_dir(tmpdirname)  # block reading files in `.cache`
            loaded_offline_predictor = MultiModalPredictor.load(path=save_path_standalone)

    offline_predictions = loaded_offline_predictor.predict(dataset.test_df, as_pandas=False)
    del loaded_offline_predictor

    # check if save with standalone=True coincide with standalone=False
    npt.assert_equal(online_predictions, offline_predictions)


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {
            "model.names": ["timm_image_0", "timm_image_1", "fusion_mlp"],
            "model.timm_image_0.checkpoint_name": "mobilenetv3_small_100",
            "model.timm_image_1.checkpoint_name": "mobilenetv3_small_100",
        },
        {
            "model.names": ["hf_text_abc", "hf_text_def", "hf_text_xyz", "fusion_mlp_123"],
            "model.hf_text_def.checkpoint_name": "monsoon-nlp/hindi-bert",
            "model.hf_text_xyz.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            "model.hf_text_abc.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2",
        },
        {
            "model.names": ["timm_image_haha", "hf_text_hello", "numerical_mlp_456", "fusion_mlp"],
            "model.timm_image_haha.checkpoint_name": "swin_tiny_patch4_window7_224",
            "model.hf_text_hello.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            "data.numerical.convert_to_text": False,
        },
    ],
)
def test_customizing_model_names(
    hyperparameters,
):
    dataset = ALL_DATASETS["petfinder"]
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters.update(
        {
            "env.num_workers": 0,
            "env.num_workers_evaluation": 0,
        }
    )
    hyperparameters_gt = copy.deepcopy(hyperparameters)
    if isinstance(hyperparameters_gt["model.names"], str):
        hyperparameters_gt["model.names"] = OmegaConf.from_dotlist([f'names={hyperparameters["model.names"]}']).names

    save_path = os.path.join(get_home_dir(), "outputs", "petfinder")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=20,
        save_path=save_path,
    )

    assert sorted(predictor._learner._config.model.names) == sorted(hyperparameters_gt["model.names"])
    for per_name in hyperparameters_gt["model.names"]:
        assert hasattr(predictor._learner._config.model, per_name)

    score = predictor.evaluate(dataset.test_df)
    verify_predictor_save_load(predictor, dataset.test_df)

    # Test for continuous fit
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=20,
    )
    assert sorted(predictor._learner._config.model.names) == sorted(hyperparameters_gt["model.names"])
    for per_name in hyperparameters_gt["model.names"]:
        assert hasattr(predictor._learner._config.model, per_name)
    verify_predictor_save_load(predictor, dataset.test_df)

    # Saving to folder, loading the saved model and call fit again (continuous fit)
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictor = MultiModalPredictor.load(root)
        predictor.fit(
            train_data=dataset.train_df,
            hyperparameters=hyperparameters,
            time_limit=10,
        )
        assert sorted(predictor._learner._config.model.names) == sorted(hyperparameters_gt["model.names"])
        for per_name in hyperparameters_gt["model.names"]:
            assert hasattr(predictor._learner._config.model, per_name)


def test_modifying_duplicate_model_names():
    dataset = ALL_DATASETS["petfinder"]
    metric_name = dataset.metric

    teacher_predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )

    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }

    teacher_predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=1,
    )
    student_predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    student_predictor.fit(
        train_data=dataset.train_df,
        time_limit=0,
    )

    teacher_predictor._learner = modify_duplicate_model_names(
        learner=teacher_predictor._learner,
        postfix="teacher",
        blacklist=student_predictor._learner._config.model.names,
    )

    # verify teacher and student have no duplicate model names
    assert all(
        [
            n not in teacher_predictor._learner._config.model.names
            for n in student_predictor._learner._config.model.names
        ]
    ), (
        f"teacher model names {teacher_predictor._learner._config.model.names} and"
        f" student model names {student_predictor._learner._config.model.names} have duplicates."
    )

    # verify each model name prefix is valid
    assert teacher_predictor._learner._model.prefix in teacher_predictor._learner._config.model.names
    if isinstance(teacher_predictor._learner._model.model, nn.ModuleList):
        for per_model in teacher_predictor._learner._model.model:
            assert per_model.prefix in teacher_predictor._learner._config.model.names

    # verify each data processor's prefix is valid
    for per_modality_processors in teacher_predictor._learner._data_processors.values():
        for per_processor in per_modality_processors:
            assert per_processor.prefix in teacher_predictor._learner._config.model.names


def test_image_bytearray():
    download_dir = "./"
    train_data_1, test_data_1 = shopee_dataset(download_dir=download_dir)
    train_data_2, test_data_2 = shopee_dataset(download_dir=download_dir, is_bytearray=True)
    predictor_1 = MultiModalPredictor(
        label="label",
    )
    predictor_2 = MultiModalPredictor(
        label="label",
    )
    model_names = ["timm_image"]
    hyperparameters = {
        "optimization.max_epochs": 2,
        "model.names": model_names,
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
    }
    predictor_1.fit(
        train_data=train_data_1,
        hyperparameters=hyperparameters,
        seed=42,
    )
    predictor_2.fit(
        train_data=train_data_2,
        hyperparameters=hyperparameters,
        seed=42,
    )

    score_1 = predictor_1.evaluate(test_data_1)
    score_2 = predictor_2.evaluate(test_data_2)
    # train and predict using different image types
    score_3 = predictor_1.evaluate(test_data_2)
    score_4 = predictor_2.evaluate(test_data_1)

    prediction_1 = predictor_1.predict(test_data_1, as_pandas=False)
    prediction_2 = predictor_2.predict(test_data_2, as_pandas=False)
    prediction_3 = predictor_1.predict(test_data_2, as_pandas=False)
    prediction_4 = predictor_2.predict(test_data_1, as_pandas=False)

    prediction_prob_1 = predictor_1.predict_proba(test_data_1, as_pandas=False)
    prediction_prob_2 = predictor_2.predict_proba(test_data_2, as_pandas=False)
    prediction_prob_3 = predictor_1.predict_proba(test_data_2, as_pandas=False)
    prediction_prob_4 = predictor_1.predict_proba(test_data_1, as_pandas=False)

    npt.assert_array_equal([score_1, score_2, score_3, score_4], [score_1] * 4)
    npt.assert_array_equal([prediction_1, prediction_2, prediction_3, prediction_4], [prediction_1] * 4)
    npt.assert_array_equal(
        [prediction_prob_1, prediction_prob_2, prediction_prob_3, prediction_prob_4], [prediction_prob_1] * 4
    )


def test_fit_with_data_path():
    download_dir = "./"
    train_csv_file = "shopee_train_data.csv"
    train_data, _ = shopee_dataset(download_dir=download_dir)
    train_data.to_csv(train_csv_file)
    predictor = MultiModalPredictor(label="label")
    predictor.fit(train_data=train_csv_file, time_limit=0)
    predictor.fit(train_data=train_csv_file, tuning_data=train_csv_file, time_limit=0)


def test_load_ckpt():
    download_dir = "./"
    train_data, test_data = shopee_dataset(download_dir=download_dir)
    predictor = MultiModalPredictor(label="label")
    predictor.fit(train_data=train_data, time_limit=20)
    src_file = os.path.join(predictor.path, "model.ckpt")
    dest_file = os.path.join(predictor.path, "epoch=8-step=18.ckpt")
    shutil.copy(src_file, dest_file)
    loaded_predictor = MultiModalPredictor.load(path=dest_file)

    predictions = predictor.predict(test_data, as_pandas=False)
    predictions2 = loaded_predictor.predict(test_data, as_pandas=False)
    npt.assert_equal(predictions, predictions2)
    predictions_prob = predictor.predict_proba(test_data, as_pandas=False)
    predictions2_prob = loaded_predictor.predict_proba(test_data, as_pandas=False)
    npt.assert_equal(predictions_prob, predictions2_prob)


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {
            "model.names": ["timm_image"],
            "model.timm_image.checkpoint_name": "mobilenetv3_small_100",
        },
        {
            "model.names": ["hf_text"],
            "model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2",
        },
        {
            "model.names": ["timm_image_haha", "hf_text_hello", "numerical_mlp_456", "fusion_mlp"],
            "model.timm_image_haha.checkpoint_name": "swin_tiny_patch4_window7_224",
            "model.hf_text_hello.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
            "data.numerical.convert_to_text": False,
        },
    ],
)
def test_hyperparameters_consistency(hyperparameters):
    dataset = ALL_DATASETS["petfinder"]
    metric_name = dataset.metric

    # pass hyperparameters to init()
    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
        hyperparameters=hyperparameters,
    )
    predictor.fit(dataset.train_df, time_limit=10)

    # pass hyperparameters to fit()
    predictor_2 = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    predictor_2.fit(
        dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=10,
    )
    assert predictor._learner._config == predictor_2._learner._config


def test_predictor_cpu_only():
    dataset = ALL_DATASETS["petfinder"]
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": ["ft_transformer"],
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "data.categorical.convert_to_text": False,  # ensure the categorical model is used.
        "data.numerical.convert_to_text": False,  # ensure the numerical model is used.
        "env.accelerator": "cpu",
    }
    predictor.fit(
        dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=10,
    )
    predictor.evaluate(dataset.test_df)
    predictor.predict(dataset.test_df)
