import os
import numpy as np



def quantile_transformer_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].type.shape[0]
    output_type = input_type([input_dim, op.quantiles_.shape[1]])
    operator.outputs[0].type = output_type


def quantile_transformer_converter(scope, operator, container):
    from scipy.stats import norm
    from skl2onnx.algebra.onnx_ops import (
        OnnxSub,
        OnnxAbs,
        OnnxSplit,
        OnnxGatherElements,
    )
    from skl2onnx.algebra.onnx_ops import OnnxArgMin, OnnxConcat, OnnxMatMul, OnnxReshape
    from skl2onnx.common.data_types import guess_numpy_type

    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    # We retrieve the unique input.
    X = operator.inputs[0]

    # In most case, computation happen in floats.
    # But it might be with double. ONNX is very strict
    # about types, every constant should have the same
    # type as the input.
    dtype = guess_numpy_type(X.type)
    batch_size = X.type.shape[0]
    n_quantiles = op.n_quantiles_

    # We tell in ONNX language how to compute the unique output.
    # op_version=opv tells which opset is requested
    C = op.quantiles_.astype(dtype)
    C_col = OnnxSplit(C, axis=1, output_names=[f"C_col{x}" for x in range(op.n_features_in_)])
    C_col.add_to(scope, container)
    X_col = OnnxSplit(X, axis=1, output_names=[f"X_col{x}" for x in range(op.n_features_in_)])
    X_col.add_to(scope, container)
    Y_col = []
    for feature_idx in range(op.n_features_in_):
        # This implements
        # > X_col[isfinite_mask] = np.interp(X_col_finite, self.references_, quantiles)
        #
        # Specifically, for yi = np.interp(xi, x, y), we implement
        # > def nearest_interp(xi, x, y):
        # >     idx = np.abs(x - xi[:,None])
        # >     return y[idx.argmin(axis=1)]
        # See https://stackoverflow.com/questions/21002799/extraploation-with-nearest-method-in-python
        repeat = OnnxMatMul(
            OnnxReshape(X_col.outputs[feature_idx], np.array([batch_size, 1])),
            np.ones(shape=(1, n_quantiles)).astype(dtype),
        )
        sub = OnnxSub(
            repeat,
            OnnxReshape(C_col.outputs[feature_idx], np.array([1, n_quantiles])),
            op_version=opv,
            output_names=[f"sub_col{feature_idx}"],
        )
        idx = OnnxAbs(sub, op_version=opv, output_names=[f"idx_col{feature_idx}"])
        argmin = OnnxArgMin(
            OnnxReshape(idx, np.array([batch_size, n_quantiles])),
            axis=1,
            op_version=opv,
            output_names=[f"argmin_col{feature_idx}"],
        )
        references = np.clip(norm.ppf(op.references_), -5.2, 5.2).astype(dtype)
        cst = np.broadcast_to(references, (batch_size, n_quantiles))
        argmin_reshaped = OnnxReshape(argmin, np.array([batch_size, 1]), output_names=[f"reshape_col{feature_idx}"])
        ref = OnnxGatherElements(
            cst, argmin_reshaped, axis=1, op_version=opv, output_names=[f"gathernd_col{feature_idx}"]
        )
        Y_col.append(OnnxReshape(ref, np.array([batch_size, 1]), output_names=[f"Y_col{feature_idx}"]))
    Y = OnnxConcat(*Y_col, axis=1, op_version=opv, output_names=out[:1])
    Y.add_to(scope, container)


def ordinal_handle_unknown_transformer_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].type.shape[0]
    output_type = input_type([input_dim, len(op.categories_len_)])
    operator.outputs[0].type = output_type


def ordinal_handle_unknown_transformer_converter(scope, operator, container):
    from skl2onnx.algebra.onnx_ops import (
        OnnxSub,
        OnnxAbs,
        OnnxSplit,
    )
    from skl2onnx.algebra.onnx_ops import OnnxArgMin, OnnxConcat, OnnxMatMul, OnnxReshape
    from skl2onnx.common.data_types import guess_numpy_type

    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    # We retrieve the unique input.
    X = operator.inputs[0]

    # In most case, computation happen in floats.
    # But it might be with double. ONNX is very strict
    # about types, every constant should have the same
    # type as the input.
    dtype = guess_numpy_type(X.type)
    batch_size = X.type.shape[0]
    num_categories = len(op.categories_)
    name_prefix = "ordinal_handle_unknown_"

    C_col = op.categories_
    X_col = OnnxSplit(X, axis=1, output_names=[f"{name_prefix}X_col{x}" for x in range(num_categories)])
    X_col.add_to(scope, container)
    Y_col = []
    for feature_idx in range(num_categories):
        # This implements
        # > X_col = np.searchsorted(X_col_finite, self.references_)
        #
        # Specifically, for yi = np.searchsorted(xi, x), we implement
        # > def searchsorted(xi, x, y):
        # >     idx = np.abs(x - xi[:,None])
        # >     return idx.argmin(axis=1)
        num_classes = len(C_col[feature_idx])
        repeat = OnnxMatMul(
            OnnxReshape(X_col.outputs[feature_idx], np.array([batch_size, 1])),
            np.ones(shape=(1, num_classes)).astype(dtype),
        )
        sub = OnnxSub(
            repeat,
            OnnxReshape(C_col[feature_idx].astype(dtype), np.array([1, num_classes])),
            op_version=opv,
            output_names=[f"{name_prefix}sub_col{feature_idx}"],
        )
        idx = OnnxAbs(sub, op_version=opv, output_names=[f"{name_prefix}idx_col{feature_idx}"])
        argmin = OnnxArgMin(
            OnnxReshape(idx, np.array([batch_size, num_classes])),
            axis=1,
            op_version=opv,
            output_names=[f"{name_prefix}argmin_col{feature_idx}"],
        )
        Y_col.append(OnnxReshape(argmin, np.array([batch_size, 1]), output_names=[f"{name_prefix}Y_col{feature_idx}"]))
    Y = OnnxConcat(*Y_col, axis=1, op_version=opv, output_names=out[:1])
    Y.add_to(scope, container)


class InferenceSessionWrapper:
    """
    Wrap around InferenceSession in onnxruntime, since it cannot be pickled.
    See https://github.com/microsoft/onnxruntime/issues/10097
    """

    def __init__(self, onnx_bytes):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_bytes.SerializeToString(), providers=["CPUExecutionProvider"])

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


class TabularNeuralNetTorchOnnxTransformer:
    def __init__(self, model):
        self.sess = InferenceSessionWrapper(model)
        self.batch_size = self.sess.get_inputs()[0].shape[0]

    def transform(self, X):
        """Run the model with the input and return the result."""
        X.columns = [c.replace("-", "_") for c in X.columns]
        inputs = self.sess.get_inputs()
        input_names = [x.name for x in inputs]
        input_dict = {}
        input_arr = X[input_names].to_numpy().astype(np.float32)
        input_size = input_arr.shape[0]
        inputs = []
        if input_size > self.batch_size:
            indices = list(np.arange(self.batch_size, input_size, self.batch_size)) + [input_size]
            inputs = np.split(input_arr, indices)
        else:
            inputs = [input_arr]
        outputs = []
        for input_arr in inputs:
            input_size = input_arr.shape[0]
            if input_size < self.batch_size:
                # padding
                pad_size = self.batch_size - input_size
                pad_shape = list(input_arr.shape)
                pad_shape[0] = pad_size
                pad_arr = np.zeros(shape=tuple(pad_shape), dtype=np.float32)
                input_arr = np.concatenate([input_arr, pad_arr])
            for idx, name in enumerate(input_names):
                input_dict[name] = input_arr[:, idx].reshape(self.batch_size, 1)
            label_name = self.sess.get_outputs()[0].name
            output_arr = self.sess.run([label_name], input_dict)[0]
            if input_size < self.batch_size:
                # remove padding
                output_arr = output_arr[:input_size, :]
            outputs.append(output_arr)
        outputs = np.concatenate(outputs)
        return outputs


class TabularNeuralNetTorchOnnxCompiler:
    name = "onnx"
    save_in_pkl = False

    @staticmethod
    def can_compile():
        """Verify whether the required package has been installed."""
        try:
            import skl2onnx
            import onnxruntime

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
        """
        if isinstance(model, TabularNeuralNetTorchOnnxTransformer):
            return model
        from sklearn.pipeline import Pipeline
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx import update_registered_converter
        from sklearn.preprocessing import QuantileTransformer
        from ..utils.categorical_encoders import OrdinalMergeRaresHandleUnknownEncoder

        update_registered_converter(
            QuantileTransformer,
            "SklearnQuantileTransformer",
            quantile_transformer_shape_calculator,
            quantile_transformer_converter,
        )
        update_registered_converter(
            OrdinalMergeRaresHandleUnknownEncoder,
            "OrdinalMergeRaresHandleUnknownEncoder",
            ordinal_handle_unknown_transformer_shape_calculator,
            ordinal_handle_unknown_transformer_converter,
        )

        if input_types is None or not isinstance(input_types[0], tuple):
            raise RuntimeError(
                "input_types argument should contain at least one tuple, e.g. [((1, 14), np.float32)]"
            )
        pipeline = Pipeline(
            steps=[
                ("processor", model[0]),
            ]
        )

        for idx, input_type in enumerate(input_types):
            input_types[idx] = (input_type[0], FloatTensorType(input_type[1]))
        onnx_model = convert_sklearn(pipeline, initial_types=input_types)

        predictor = TabularNeuralNetTorchOnnxTransformer(model=onnx_model)
        TabularNeuralNetTorchOnnxCompiler.save(onnx_model, path)
        return predictor

    @staticmethod
    def save(model, path: str) -> str:
        """Save the compiled model into onnx file format."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(os.path.join(path, "model.onnx"), "wb") as f:
            f.write(model.SerializeToString())
        return os.path.join(path, "model.onnx")

    @staticmethod
    def load(path: str) -> TabularNeuralNetTorchOnnxTransformer:
        """Load from the path that contains an onnx file."""
        import onnx

        onnx_bytes = onnx.load(os.path.join(path, "model.onnx"))
        return TabularNeuralNetTorchOnnxTransformer(model=onnx_bytes)
