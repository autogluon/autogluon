import os
import numpy as np


class InferenceSessionWrapper:
    """
    Wrap around InferenceSession in onnxruntime, since it cannot be pickled.
    See https://github.com/microsoft/onnxruntime/issues/10097
    """
    def __init__(self, onnx_bytes):
        import onnxruntime as rt
        self.sess = rt.InferenceSession(onnx_bytes.SerializeToString(),
                                        providers=['CPUExecutionProvider'])

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
    def __init__(self, model, num_classes):
        self.num_classes = num_classes
        self.sess = InferenceSessionWrapper(model)
        self.model = model

    def predict(self, X):
        input_name = self.sess.get_inputs()[0].name
        label_name = self.sess.get_outputs()[0].name
        return self.sess.run([label_name], {input_name: X})[0].squeeze()

    def predict_proba(self, X):
        input_name = self.sess.get_inputs()[0].name
        label_name = self.sess.get_outputs()[1].name
        pred_proba = self.sess.run([label_name], {input_name: X})[0]
        pred_proba = np.array([[r[i] for i in range(self.num_classes)] for r in pred_proba])
        return pred_proba


class RFOnnxCompiler:
    name = 'onnx'
    save_in_pkl = False

    @staticmethod
    def can_compile():
        try:
            import skl2onnx
            return True
        except ImportError:
            return False

    @staticmethod
    def compile(model, input_types=None):
        if input_types is None or not isinstance(input_types[0], tuple):
            raise RuntimeError("input_types argument should contain at least one tuple"
                               ", e.g. [((1, 14), np.float32)]")
        if isinstance(model, RFOnnxPredictor):
            return model
        # Convert into ONNX format
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        input_shape = list(input_types[0][0])
        initial_type = [('float_input', FloatTensorType(input_shape))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        predictor =  RFOnnxPredictor(model=onnx_model, num_classes=model.n_classes_)
        return predictor

    @staticmethod
    def save(model, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + "model.onnx", "wb") as f:
            f.write(model.SerializeToString())
        return RFOnnxCompiler.load(path=path)

    @staticmethod
    def load(obj, path: str):
        import onnx
        onnx_bytes = onnx.load(path + "model.onnx")
        model = InferenceSessionWrapper(onnx_bytes)
        return RFOnnxPredictor(model=model)
