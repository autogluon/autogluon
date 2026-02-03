import os

import numpy as np


class InferenceSessionWrapper:
    """
    Wrap around InferenceSession in onnxruntime, since it cannot be pickled.
    See https://github.com/microsoft/onnxruntime/issues/10097
    """

    def __init__(self, onnx_bytes):
        import onnxruntime as rt

        self.sess = rt.InferenceSession(onnx_bytes.SerializeToString(), providers=["CPUExecutionProvider"])

    def run(self, *args):
        return self.sess.run(*args)

    def get_inputs(self, *args):
        return self.sess.get_inputs(*args)

    def get_outputs(self, *args):
        return self.sess.get_outputs(*args)

    def __getstate__(self):
        # No need to duplicate the model parameters here.
        return {}

    def __setstate__(self, values):
        pass


class RFOnnxPredictor:
    def __init__(self, model):
        self.sess = InferenceSessionWrapper(model)
        self.num_classes = self.sess.get_outputs()[-1].shape[1]

    def predict(self, X):
        """Run the model with the input and return the result."""
        input_name = self.sess.get_inputs()[0].name
        label_name = self.sess.get_outputs()[0].name
        return self.sess.run([label_name], {input_name: X})[0].squeeze()

    def predict_proba(self, X):
        """Run the model with the input, and return probabilities as result."""
        input_name = self.sess.get_inputs()[0].name
        label_name = self.sess.get_outputs()[1].name
        pred_proba = self.sess.run([label_name], {input_name: X})[0]
        pred_proba = np.array([[r[i] for i in range(self.num_classes)] for r in pred_proba])
        return pred_proba


class RFOnnxCompiler:
    name = "onnx"
    save_in_pkl = False

    @staticmethod
    def can_compile():
        """Verify whether the required package has been installed."""
        try:
            import onnxruntime
            import skl2onnx

            return True
        except ImportError:
            return False

    @staticmethod
    def compile(model, path: str, input_types=None):
        """
        Compile the trained model for faster inference.

        Parameters
        ----------
        model
            The native model that is expected to be compiled.
        path : str
            The path for saving the compiled model.
        input_types : list, default=None
            A list of tuples containing shape and element type info, e.g. [((1, 14), np.float32),].
            The list would be used as the input data for the model.
            The compiler would optimize the model to perform best with the given input type.
        """
        if input_types is None or not isinstance(input_types[0], tuple):
            raise RuntimeError("input_types argument should contain at least one tuple, e.g. [((1, 14), np.float32)]")
        if isinstance(model, RFOnnxPredictor):
            return model

        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

        input_shape = list(input_types[0][0])
        initial_type = [("float_input", FloatTensorType(input_shape))]

        # Without ZipMap
        # See http://onnx.ai/sklearn-onnx/auto_examples/plot_convert_zipmap.html#without-zipmap
        options = {}
        if isinstance(model, (RandomForestClassifier, ExtraTreesClassifier)):
            options = {id(model): {"zipmap": False}}

        # Convert the model to onnx
        onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)
        predictor = RFOnnxPredictor(model=onnx_model)
        RFOnnxCompiler.save(onnx_model, path)
        return predictor

    @staticmethod
    def save(model, path: str) -> str:
        """Save the compiled model into onnx file format."""
        file_path = os.path.join(path, "model.onnx")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(model.SerializeToString())
        return os.path.join(path, "model.onnx")

    @staticmethod
    def load(path: str) -> RFOnnxPredictor:
        """Load from the path that contains an onnx file."""
        import onnx

        onnx_bytes = onnx.load(os.path.join(path, "model.onnx"))
        return RFOnnxPredictor(model=onnx_bytes)
