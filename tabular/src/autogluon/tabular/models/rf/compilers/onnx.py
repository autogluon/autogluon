import os
import numpy as np

class RFOnnxPredictor:
    def __init__(self, obj, model):
        self.num_classes = obj.num_classes
        self.model = model

    def predict(self, X):
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        return self.model.run([label_name], {input_name: X})[0].squeeze()

    def predict_proba(self, X):
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[1].name
        pred_proba = self.model.run([label_name], {input_name: X})[0]
        pred_proba = np.array([[r[i] for i in range(self.num_classes)] for r in pred_proba])
        return pred_proba


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
    def compile(obj, path: str):
        if isinstance(obj.model, RFOnnxPredictor):
            return obj.model
        # Convert into ONNX format
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [('float_input', FloatTensorType([None, obj._num_features_post_process]))]
        onx = convert_sklearn(obj.model, initial_types=initial_type)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + "model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        return RFOnnxCompiler.load(obj=obj, path=path)

    @staticmethod
    def load(obj, path: str):
        import onnx
        onnx_bytes = onnx.load(path + "model.onnx")
        model = InferenceSessionWrapper(onnx_bytes)
        return RFOnnxPredictor(obj=obj, model=model)
