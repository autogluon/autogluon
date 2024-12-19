import os
import shutil

from autogluon.multimodal import MultiModalPredictor

from ..utils import PetFinderDataset, get_home_dir, verify_predictor_save_load


def test_ensemble_from_scratch():
    dataset = PetFinderDataset()
    metric_name = dataset.metric

    hyperparameters = {
        "learner_names": ["lf_mlp", "lf_clip"],
        "lf_mlp": {
            "model.names": ["ft_transformer", "timm_image", "hf_text", "fusion_mlp"],
            "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"],
            "env.num_workers": 0,
            "env.num_workers_inference": 0,
            "optim.max_epochs": 1,
        },
        "lf_clip": {
            "model.names": ["ft_transformer", "clip_image", "clip_text", "fusion_mlp"],
            "model.clip_image.data_types": ["image"],
            "model.clip_text.data_types": ["text"],
            "env.num_workers": 0,
            "env.num_workers_inference": 0,
            "optim.max_epochs": 1,
        },
    }
    save_path = os.path.join(get_home_dir(), "outputs", "ensemble")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
        use_ensemble=True,
        ensemble_mode="one_shot",
    )
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=20,
        save_path=save_path,
    )
    assert len(predictor._learner._all_learners) == 2
    assert len(predictor._learner._selected_learners) == 2
    score = predictor.evaluate(dataset.test_df)
    verify_predictor_save_load(predictor, dataset.test_df, verify_embedding=False)


def test_ensemble_with_pretrained_predictors():
    dataset = PetFinderDataset()
    metric_name = dataset.metric

    hyperparameters_1 = {
        "model.names": ["ft_transformer", "timm_image", "hf_text", "fusion_mlp"],
        "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"],
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
        "optim.max_epochs": 1,
    }
    hyperparameters_2 = {
        "model.names": ["ft_transformer", "clip_image", "clip_text", "fusion_mlp"],
        "model.clip_image.data_types": ["image"],
        "model.clip_text.data_types": ["text"],
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
        "optim.max_epochs": 1,
    }

    save_path_1 = os.path.join(get_home_dir(), "outputs", "lf_mlp")
    save_path_2 = os.path.join(get_home_dir(), "outputs", "lf_clip")
    save_path_ensemble = os.path.join(get_home_dir(), "outputs", "ensemble")

    for per_path in [save_path_1, save_path_2, save_path_ensemble]:
        if os.path.exists(per_path):
            shutil.rmtree(per_path)

    predictors = []
    for per_path, per_hparams in zip([save_path_1, save_path_2], [hyperparameters_1, hyperparameters_2]):
        per_predictor = MultiModalPredictor(
            label=dataset.label_columns[0],
            problem_type=dataset.problem_type,
            eval_metric=metric_name,
        )
        per_predictor.fit(
            train_data=dataset.train_df,
            hyperparameters=per_hparams,
            time_limit=20,
            save_path=per_path,
        )
        predictors.append(per_predictor.path)

    ensemble_predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
        use_ensemble=True,
        ensemble_mode="one_shot",
    )
    ensemble_predictor.fit(
        train_data=dataset.train_df,
        predictors=predictors,
    )
    assert len(ensemble_predictor._learner._all_learners) == 2
    assert len(ensemble_predictor._learner._selected_learners) == 2
    score = ensemble_predictor.evaluate(dataset.test_df)
    verify_predictor_save_load(ensemble_predictor, dataset.test_df, verify_embedding=False)
