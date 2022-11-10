import pandas as pd
import pytest
from ray import tune

from autogluon.core.hpo.ray_tune_constants import SCHEDULER_PRESETS, SEARCHER_PRESETS
from autogluon.multimodal import MultiModalPredictor


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
    [("microsoft/deberta-v3-small", None, None), ("google/electra-small-discriminator", "bayes", "FIFO")],
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
        time_limit=40,
        hyperparameters={"model.ner.checkpoint_name": checkpoint_name, "optimization.learning_rate": lr},
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )

    scores = predictor.evaluate(train_data)

    test_predict = train_data.drop(label_col, axis=1)
    test_predict = test_predict.head(2)
    predictions = predictor.predict(test_predict)
    embeddings = predictor.extract_embedding(test_predict)
