import tempfile
import copy
import pickle

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import (
    MODEL,
    DATA,
    OPTIMIZATION,
    ENVIRONMENT,
    DISTILLER,
    BINARY,
    MULTICLASS,
    UNIFORM_SOUP,
    GREEDY_SOUP,
    BEST,
    NORM_FIT,
    BIT_FIT,
    LORA,
    LORA_BIAS,
    LORA_NORM,
)
from datasets import (
    PetFinderDataset,
    HatefulMeMesDataset,
    AEDataset,
)
from test_predictor import verify_predictor_save_load

ALL_DATASETS = {
    "petfinder": PetFinderDataset,
    "hateful_memes": HatefulMeMesDataset,
    "ae": AEDataset,
}


def test_mixup():
    dataset = ALL_DATASETS["petfinder"]()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    hyperparameters = {
        "optimization.max_epochs": 1,
        "optimization.top_k_average_method": BEST,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "data.mixup.turn_on": True,
    }

    with tempfile.TemporaryDirectory() as save_path:
        predictor.fit(
            train_data=dataset.train_df,
            config=config,
            time_limit=30,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

        score = predictor.evaluate(dataset.test_df)
        verify_predictor_save_load(predictor, dataset.test_df)


def test_textagumentor_deepcopy():
    dataset = ALL_DATASETS["ae"]()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    hyperparameters = {
        "optimization.max_epochs": 1,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "model.hf_text.text_trivial_aug_maxscale": 0.05,
        "model.hf_text.text_train_augment_types": ["identity"],
        "optimization.top_k_average_method": "uniform_soup",
    }

    with tempfile.TemporaryDirectory() as save_path:
        predictor.fit(
            train_data=dataset.train_df,
            config=config,
            time_limit=10,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

    # Deepcopy data processors
    predictor._data_processors = copy.deepcopy(predictor._data_processors)

    # Test copied data processors
    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=10,
    )

    # Copy data processors via pickle + load
    predictor._data_processors = pickle.loads(pickle.dumps(predictor._data_processors))

    # Test copied data processors
    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=10,
    )


def test_trivialaugment():
    dataset = ALL_DATASETS["petfinder"]()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    hyperparameters = {
        "optimization.max_epochs": 1,
        "optimization.top_k_average_method": BEST,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "data.mixup.turn_on": True,
        "model.hf_text.text_trivial_aug_maxscale": 0.1,
        "model.hf_text.text_aug_detect_length": 10,
        "model.timm_image.train_transform_types": ["resize_shorter_side", "center_crop", "trivial_augment"],
    }

    with tempfile.TemporaryDirectory() as save_path:
        predictor.fit(
            train_data=dataset.train_df,
            config=config,
            time_limit=30,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

        score = predictor.evaluate(dataset.test_df)
        verify_predictor_save_load(predictor, dataset.test_df)
