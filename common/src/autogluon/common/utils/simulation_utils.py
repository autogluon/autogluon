from collections import defaultdict
from typing import Any, Dict, Tuple

import pandas as pd


def _recursive_dd():
    return defaultdict(_recursive_dd)


def _dd_to_dict(dd):
    dd = dict(dd)
    for k, v in dd.items():
        if isinstance(v, defaultdict):
            dd[k] = _dd_to_dict(v)
    return dd


def convert_simulation_artifacts_to_tabular_predictions_dict(simulation_artifacts: Dict[str, Dict[int, Dict[str, Any]]]) -> Tuple[dict, dict]:
    """
    Converts raw simulation artifacts to the format required for ensemble simulation.

    Parameters
    ----------
    simulation_artifacts : Dict[str, Dict[int, Dict[str, Any]]]
        Dictionary of simulation artifacts.
        This is a nested dictionary of dataset_name -> fold -> simulation_artifact.
        simulation_artifact is a dictionary acquired for a given dataset-fold via calling `predictor.get_simulation_artifact(test_data=test_data)` post-fit.

    Returns
    -------
    aggregated_pred_proba: Nested dictionary of dataset_name -> fold -> pred_proba
        pred_proba is a dictionary with the following keys:
            "pred_proba_dict_val",  # a dictionary of validation prediction probabilities for each model
            "pred_proba_dict_test",  # a dictionary of test prediction probabilities for each model
    aggregated_ground_truth: Nested dictionary of dataset_name -> fold -> task_metadata
        task_metadata is a dictionary with the following keys:
            "y_val",  # The validation ground truth
            "y_test",  # The test ground truth
            "eval_metric",  # The name of the evaluation metric
            "problem_type",  # The problem type
            "ordered_class_labels",  # The original class labels
            "ordered_class_labels_transformed",  # The transformed class labels
            "problem_type_transform",  # The transformed problem type
            "num_classes",  # The number of classes
            "label",  # The label column name
    """
    aggregated_pred_proba = _recursive_dd()
    aggregated_ground_truth = _recursive_dd()
    for task_name in simulation_artifacts.keys():
        for fold in simulation_artifacts[task_name].keys():
            zeroshot_metadata = simulation_artifacts[task_name][fold]
            fold = int(fold)
            if fold not in aggregated_ground_truth[task_name]:
                for k in [
                    "y_val",
                    "y_test",
                    "eval_metric",
                    "problem_type",
                    "problem_type_transform",
                    "ordered_class_labels",
                    "ordered_class_labels_transformed",
                    "num_classes",
                    "label",
                ]:
                    aggregated_ground_truth[task_name][fold][k] = zeroshot_metadata[k]
            for k in ["pred_proba_dict_val", "pred_proba_dict_test"]:
                for m, pred_proba in zeroshot_metadata[k].items():
                    if aggregated_ground_truth[task_name][fold]["problem_type"] == "binary":
                        if isinstance(pred_proba, pd.DataFrame):
                            assert len(pred_proba.columns) == 2
                            pred_proba = pred_proba[1]
                        assert isinstance(pred_proba, pd.Series)
                    aggregated_pred_proba[task_name][fold][k][m] = pred_proba
    aggregated_pred_proba = _dd_to_dict(aggregated_pred_proba)
    aggregated_ground_truth = _dd_to_dict(aggregated_ground_truth)
    return aggregated_pred_proba, aggregated_ground_truth
