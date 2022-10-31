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
    def __init__(self, model):
        # self.num_classes = num_classes
        self.sess = InferenceSessionWrapper(model)
        self.num_classes = self.sess.get_outputs()[1].shape[1]

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
    def compile(model, path: str, input_types=None):
        if input_types is None or not isinstance(input_types[0], tuple):
            raise RuntimeError("input_types argument should contain at least one tuple"
                               ", e.g. [((1, 14), np.float32)]")
        if isinstance(model, RFOnnxPredictor):
            return model

        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        input_shape = list(input_types[0][0])
        initial_type = [('float_input', FloatTensorType(input_shape))]
        # Without ZipMap
        # See http://onnx.ai/sklearn-onnx/auto_examples/plot_convert_zipmap.html#without-zipmap
        options = {id(model): {'zipmap': False}}

        # Convert the model to onnx
        onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)
        predictor =  RFOnnxPredictor(model=onnx_model)
        RFOnnxCompiler.save(onnx_model, path)
        return predictor

    @staticmethod
    def save(model, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + "model.onnx", "wb") as f:
            f.write(model.SerializeToString())
        return path + "model.onnx"

    @staticmethod
    def load(path: str):
        import onnx
        onnx_bytes = onnx.load(path + "model.onnx")
        return RFOnnxPredictor(model=onnx_bytes)
