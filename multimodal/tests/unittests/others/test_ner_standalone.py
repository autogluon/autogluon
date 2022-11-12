import os
import shutil
import tempfile
from unittest import mock

import numpy.testing as npt
import pandas as pd
import torch

from autogluon.multimodal import MultiModalPredictor

from .utils import get_home_dir


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
            torch.hub.set_dir(tmpdirname)  # block reading files in `.cache`
            loaded_offline_predictor = MultiModalPredictor.load(path=save_path_standalone)

    offline_predictions = loaded_offline_predictor.predict(test_data, as_pandas=False)
    del loaded_offline_predictor
    npt.assert_equal(online_predictions[0], offline_predictions[0])
