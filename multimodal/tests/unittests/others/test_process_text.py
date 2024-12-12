import copy
import os
import pickle
import shutil
import tempfile

import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.data.process_text import TextProcessor

from ..utils import AEDataset


def test_text_processor_deepcopy_and_dump():
    dataset = AEDataset()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "optim.max_epochs": 1,
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model.hf_text.text_trivial_aug_maxscale": 0.05,
        "model.hf_text.text_train_augment_types": ["identity"],
        "optim.top_k_average_method": "uniform_soup",
    }

    with tempfile.TemporaryDirectory() as save_path:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        predictor.fit(
            train_data=dataset.train_df,
            time_limit=10,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

    # Deepcopy data processors
    predictor._learner._data_processors = copy.deepcopy(predictor._learner._data_processors)

    # Test copied data processors
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=10,
    )

    # Copy data processors via pickle + load
    predictor._learner._data_processors = pickle.loads(pickle.dumps(predictor._learner._data_processors))

    # Test copied data processors
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=10,
    )


@pytest.mark.parametrize(
    "lengths,max_length,do_merge,gt_trimmed_lengths",
    [
        ([7, 8, 9], 29, True, [7, 8, 9]),
        ([7, 8, 9], 24, True, [7, 8, 9]),
        ([7, 8, 9], 23, True, [7, 8, 8]),
        ([7, 8, 9], 22, True, [7, 8, 7]),
        ([7, 8, 9], 21, True, [7, 7, 7]),
        ([7, 8, 9], 20, True, [7, 7, 6]),
        ([7, 8, 9], 19, True, [7, 6, 6]),
        ([7, 8, 9], 18, True, [6, 6, 6]),
        ([7, 8, 9], 10, False, [7, 8, 9]),
        ([7, 8, 9], 9, False, [7, 8, 9]),
        ([7, 8, 9], 8, False, [7, 8, 8]),
        ([7, 8, 9], 7, False, [7, 7, 7]),
        ([7, 8, 9], 6, False, [6, 6, 6]),
    ],
)
def test_trim_token_sequence(lengths, max_length, do_merge, gt_trimmed_lengths):
    trimmed_lengths = TextProcessor.get_trimmed_lengths(
        lengths=lengths,
        max_length=max_length,
        do_merge=do_merge,
    )
    assert trimmed_lengths == gt_trimmed_lengths
