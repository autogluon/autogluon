import os
import shutil
import tempfile

import numpy.testing as npt
import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import (
    BEST,
    BINARY,
    BIT_FIT,
    DATA,
    DISTILLER,
    ENV,
    GREEDY_SOUP,
    IA3,
    IMAGE_BASE64_STR,
    IMAGE_BYTEARRAY,
    LORA,
    LORA_BIAS,
    LORA_NORM,
    MODEL,
    MULTICLASS,
    NORM_FIT,
    OPTIM,
    UNIFORM_SOUP,
)

from ..utils import (
    AEDataset,
    HatefulMeMesDataset,
    PetFinderDataset,
    get_home_dir,
    verify_no_redundant_model_configs,
    verify_predictor_realtime_inference,
    verify_predictor_save_load,
)

ALL_DATASETS = {
    "petfinder": PetFinderDataset(),
    "petfinder_bytearray": PetFinderDataset(is_bytearray=True),
    "hateful_memes": HatefulMeMesDataset(),
    "hateful_memes_bytearray": HatefulMeMesDataset(is_bytearray=True),
    "ae": AEDataset(),
}


@pytest.mark.parametrize(
    "dataset_name,model_names,text_backbone,image_backbone,top_k_average_method,peft,loss_func",
    [
        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "fusion_mlp"],
            "sentence-transformers/all-MiniLM-L6-v2",
            "swin_tiny_patch4_window7_224",
            GREEDY_SOUP,
            None,
            "auto",
        ),
        (
            "petfinder",
            ["ft_transformer", "clip_image", "clip_text", "fusion_mlp"],
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-base-patch32",
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
            ["ft_transformer", "hf_text", "fusion_transformer"],
            "sentence-transformers/all-MiniLM-L6-v2",
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
    peft,
    loss_func,
):
    dataset = ALL_DATASETS[dataset_name]
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "optim.max_epochs": 1,
        "model.names": model_names,
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
        "optim.top_k_average_method": top_k_average_method,
        "optim.peft": peft,
        "optim.loss_func": loss_func,
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
        elif "hf_text" in model_names:
            hyperparameters.update(
                {
                    "model.hf_text.checkpoint_name": text_backbone,
                }
            )
        elif "clip_text" in model_names:
            hyperparameters.update(
                {
                    "model.clip_text.checkpoint_name": text_backbone,
                }
            )
    if image_backbone is not None:
        if "timm_image" in model_names:
            hyperparameters.update(
                {
                    "model.timm_image.checkpoint_name": image_backbone,
                }
            )
        elif "clip_image" in model_names:
            hyperparameters.update(
                {
                    "model.clip_image.checkpoint_name": image_backbone,
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
    "dataset_name,model_names,text_backbone,image_backbone,top_k_average_method,peft,loss_func",
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
    peft,
    loss_func,
):
    dataset = ALL_DATASETS[dataset_name]
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "optim.max_epochs": 1,
        "model.names": model_names,
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
        "optim.top_k_average_method": top_k_average_method,
        "optim.peft": peft,
        "optim.loss_func": loss_func,
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
    verify_predictor_realtime_inference(predictor, dataset.test_df)


def test_predictor_standalone():  # test standalone feature in MultiModalPredictor.save()
    from unittest import mock

    import torch

    requests_gag = mock.patch(
        "requests.Session.request",
        mock.Mock(side_effect=RuntimeError("Please use the `responses` library to mock HTTP in your tests.")),
    )

    dataset = ALL_DATASETS["petfinder"]

    hyperparameters = {
        "optim.max_epochs": 1,
        "model.names": ["ft_transformer", "timm_image", "hf_text", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
    }

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    save_path = os.path.join(get_home_dir(), "outputs", "standalone", "false")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=30,
        save_path=save_path,
        standalone=False,
    )

    save_path_standalone = os.path.join(get_home_dir(), "outputs", "standalone", "true")
    if os.path.exists(save_path_standalone):
        shutil.rmtree(save_path_standalone)

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
