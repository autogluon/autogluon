from .cloud_predictor import CloudPredictor


class TextCloudPredictor(CloudPredictor):

    predictor_file_name = "TextCloudPredictor.pkl"

    @property
    def predictor_type(self):
        return "text"

    def _get_local_predictor_cls(self):
        from autogluon.text import TextPredictor

        predictor_cls = TextPredictor
        return predictor_cls
