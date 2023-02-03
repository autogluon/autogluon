import pandas as pd
import numpy as np

from .learner import MultiModalLearner

from ..constants import (
COLUMN_FEATURES,
Y_PRED,
Y_TRUE
)

from ..utils import (
predict,
extract_from_output,
turn_on_off_feature_column_info,
compute_score,
try_to_infer_pos_label,
init_pretrained
)


from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from typing import List, Optional, Union, Dict

class FusionSVMClassificationLearner(MultiModalLearner):
    def __init__(
            self,
            predictor,
    ):
        super().__init__()
        self.clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))

        self.predictor = predictor

    def fit(self, train_data: Union[pd.DataFrame, dict, list]):
        """
        fit the svm clf with training data
        Parameters
        ----------
        train_data: pd.DataFrame containing training data as in Predictor, or str file containing annotations
        """
        features = self.extract_embedding(data=train_data)
        labels = np.array(train_data[self.predictor._label_column])
        self.clf.fit(features, labels)

    def extract_embedding(
            self,
            data: Union[pd.DataFrame, dict, list],
            # as_tensor: Optional[bool] = False,
            # as_pandas: Optional[bool] = False,
            realtime: Optional[bool] = None,
    ):
        self.predictor.__verify_inference_ready()
        turn_on_off_feature_column_info(
            data_processors=self.predictor._data_processors,
            flag=True,
        )
        outputs = predict(
            predictor=self.predictor,
            data=data,
            requires_label=False,
            realtime=realtime,
        )

        features = extract_from_output(outputs=outputs, ret_type=COLUMN_FEATURES, as_ndarray=True)
        # if as_pandas:
        #     features = pd.DataFrame(features, index=data.index)

        # if self._problem_type in [FEATURE_EXTRACTION, ZERO_SHOT_IMAGE_CLASSIFICATION]:
        #     features = extract_from_output(outputs=outputs, ret_type=COLUMN_FEATURES, as_ndarray=as_tensor is False)
        #     if return_masks:
        #         masks = extract_from_output(outputs=outputs, ret_type=MASKS, as_ndarray=as_tensor is False)
        # else:
        #     features = extract_from_output(outputs=outputs, ret_type=FEATURES, as_ndarray=as_tensor is False)

        # if as_pandas:
        #     features = pd.DataFrame(features, index=data.index)
        #     if return_masks:
        #         masks = pd.DataFrame(masks, index=data.index)
        return features

    def predict(self, data: Union[pd.DataFrame, dict, list]):
        features = self.extract_embedding(data)
        preds = self.clf.predict(features)
        return preds

    def predict_proba(self, data: Union[pd.DataFrame, dict, list]):
        features = self.extract_embedding(data)
        preds = self.clf.predict_proba(features)
        return preds

    def evaluate(
            self,
            data: Union[pd.DataFrame, dict, list],
            metrics: Optional[Union[str, List[str]]] = None,
            return_pred: Optional[bool] = False,
            realtime: Optional[bool] = None,
    ):
        self.predictor.__verify_inference_ready()
        features = self.extract_embedding(data)
        preds = self.clf.predict(features)
        labels = np.array(data[self.predictor._label_column])

        metric_data = {}
        # TODO: Support BINARY and MULTICLASS
        metric_data.update(
            {
                Y_PRED: preds,
                Y_TRUE: labels,
            }
        )

        if self.predictor._df_preprocessor is None:
            data, df_preprocessor, data_processors = self.predictor._on_predict_start(
                data=data,
                requires_label=True,
            )
        else:
            df_preprocessor = self.predictor._df_preprocessor

        if metrics is None:
            metrics = [self.predictor._eval_metric_name]
        if isinstance(metrics, str):
            metrics = [metrics]

        results = {}
        for per_metric in metrics:
            pos_label = try_to_infer_pos_label(
                data_config=self.predictor._config.data,
                label_encoder=df_preprocessor.label_generator,
                problem_type=self.predictor._problem_type,
            )
            score = compute_score(
                metric_data=metric_data,
                metric_name=per_metric.lower(),
                pos_label=pos_label,
            )
            results[per_metric] = score

        if return_pred:
            return results, preds
        return results