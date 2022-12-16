import copy

from ..utils.ag_sagemaker import AutoGluonImageRealtimePredictor
from ..utils.constants import VALID_ACCEPT
from ..utils.utils import read_image_bytes_and_encode
from .cloud_predictor import CloudPredictor


class ImageCloudPredictor(CloudPredictor):

    predictor_file_name = "ImageCloudPredictor.pkl"

    @property
    def predictor_type(self):
        return "image"

    @property
    def _realtime_predictor_cls(self):
        return AutoGluonImageRealtimePredictor

    def _get_local_predictor_cls(self):
        from autogluon.vision import ImagePredictor

        predictor_cls = ImagePredictor
        return predictor_cls

    def fit(self, *, predictor_init_args, predictor_fit_args, image_column, **kwargs):
        super().fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            image_column=image_column,
            **kwargs,
        )

    def predict_real_time(self, test_data, accept="application/x-parquet"):
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced.
            Can be an numpy.ndarray containing path to test images
            Or a local path to a single image file.
            Or a list of local paths to image files.
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json

        Returns
        -------
        Pandas.DataFrame
        Predict results in DataFrame
        """
        assert self.endpoint, "Please call `deploy()` to deploy an endpoint first."
        assert accept in VALID_ACCEPT, f"Invalid accept type. Options are {VALID_ACCEPT}."

        import numpy as np

        if isinstance(test_data, str):
            test_data = [test_data]
        if isinstance(test_data, list):
            test_data = np.array([read_image_bytes_and_encode(image) for image in test_data], dtype="object")

        assert isinstance(test_data, np.ndarray), f"Invalid test data format {type(test_data)}"

        return self._predict_real_time(test_data=test_data, accept=accept)

    def predict(
        self,
        test_data,
        **kwargs,
    ):
        """
        test_data: str
            The test data to be inferenced. Can be a local path to a directory containing the images.
        kwargs:
            Refer to `CloudPredictor.predict()`
        """
        split_type = None
        content_type = "application/x-image"
        kwargs = copy.deepcopy(kwargs)
        transformer_kwargs = kwargs.pop("transformer_kwargs", dict())
        transformer_kwargs["strategy"] = "SingleRecord"
        super().predict(
            test_data,
            split_type=split_type,
            content_type=content_type,
            transformer_kwargs=transformer_kwargs,
            **kwargs,
        )
