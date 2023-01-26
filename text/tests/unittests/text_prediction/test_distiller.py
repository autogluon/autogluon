import os
import shutil
import numpy as np

from autogluon.core.utils.loaders import load_pd
from autogluon.text import TextPredictor
from test_predictor_pytorch import verify_predictor_save_load


def test_distillation():
    train_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/" "glue/sst/train.parquet")
    test_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/" "glue/sst/dev.parquet")
    rng_state = np.random.RandomState(123)
    train_perm = rng_state.permutation(len(train_data))
    test_perm = rng_state.permutation(len(test_data))
    train_data = train_data.iloc[train_perm[:100]]
    test_data = test_data.iloc[test_perm[:10]]

    teacher_predictor = TextPredictor(label="label", eval_metric="acc")

    hyperparameters = {
        "model.hf_text.checkpoint_name": "nlpaueb/legal-bert-small-uncased",
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }

    teacher_save_path = os.path.join("sst", "teacher")
    if os.path.exists(teacher_save_path):
        shutil.rmtree(teacher_save_path)

    teacher_predictor = teacher_predictor.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        time_limit=30,
        save_path=teacher_save_path,
    )

    # test for distillation
    predictor = TextPredictor(label="label", eval_metric="acc")

    student_save_path = os.path.join("sst", "student")
    if os.path.exists(student_save_path):
        shutil.rmtree(student_save_path)

    predictor = predictor.fit(
        train_data=train_data,
        teacher_predictor=teacher_predictor,
        hyperparameters=hyperparameters,
        time_limit=30,
        save_path=student_save_path,
    )
    verify_predictor_save_load(predictor, test_data)

    # test for distillation with teacher predictor path
    predictor = TextPredictor(label="label", eval_metric="acc")

    student_save_path = os.path.join("sst", "student")
    if os.path.exists(student_save_path):
        shutil.rmtree(student_save_path)

    predictor = predictor.fit(
        train_data=train_data,
        teacher_predictor=teacher_predictor.path,
        hyperparameters=hyperparameters,
        time_limit=30,
        save_path=student_save_path,
    )
    verify_predictor_save_load(predictor, test_data)
