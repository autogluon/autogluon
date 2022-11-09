import os
import numpy as np

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS, QUANTILE


class TabularNeuralNetTorchTvmPredictor:
    def __init__(self, model, architecture_desc : dict = None):
        if architecture_desc is None:
            architecture_desc = {}
        import tvm
        from tvm.contrib import graph_executor
        dev = tvm.cpu()
        self.module = graph_executor.GraphModule(model["default"](dev))
        self.has_vector_features = architecture_desc.get("has_vector_features", False)
        self.has_embed_features = architecture_desc.get("has_embed_features", False)
        self.problem_type = architecture_desc.get("problem_type", None)
        self.architecture_desc = architecture_desc

    def _get_input_vector(self, data_batch : dict) -> list:
        input_data = []
        if self.has_vector_features:
            input_data.append(data_batch['vector'].numpy())
        if self.has_embed_features:
            embed_data = data_batch['embed']
            input_data += [d.numpy() for d in embed_data]
        return input_data

    def predict(self, X):
        """Run the model with the input and return the result."""
        import tvm
        input_data = self._get_input_vector(X)
        num_net_outputs = self.architecture_desc['num_net_outputs']
        data_size = input_data[0].shape[0]
        batch_size = self.module._get_input("input0").shape[0]
        out_shape = (batch_size, num_net_outputs)

        # Pad the batch if input data is small
        if data_size < batch_size:
            for i, data in enumerate(input_data):
                pad_shape = list(data.shape)
                pad_shape[0] = batch_size - data_size
                pad_data = np.zeros(tuple(pad_shape))
                input_data[i] = np.concatenate([input_data[i], pad_data])
        if data_size > batch_size:
            raise RuntimeError("Input data size is larger than expected. "
                               f"Try use DataLoader with batch_size={batch_size}.")

        # Run on graph executor
        for i, v in enumerate(input_data):
            self.module.set_input(f"input{i}", v)
        self.module.run()
        predict_data = self.module.get_output(0, tvm.nd.empty(out_shape)).numpy()

        # Remove padded result from output
        if data_size < batch_size:
            predict_data = predict_data[:data_size]

        # Problem type specific post processing
        if self.problem_type == QUANTILE:
            predict_data = torch.sort(predict_data, -1)[0]  # sorting ensures monotonicity of quantile estimates
        elif self.problem_type in [BINARY, MULTICLASS, SOFTCLASS]:
            from scipy.special import softmax
            predict_data = softmax(predict_data)  # convert NN output to probability
        elif self.problem_type == REGRESSION:
            predict_data = predict_data.flatten()
        if self.problem_type == BINARY:
            predict_data = predict_data[:,1]

        return predict_data

    def predict_proba(self, X):
        """Run the model with the input, and return probabilities as result."""
        pass


class TabularNeuralNetTorchTvmCompiler:
    name = 'tvm'
    save_in_pkl = False

    @staticmethod
    def can_compile():
        """Verify whether the required package has been installed."""
        try:
            import tvm
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
        import torch
        import tvm
        from tvm import relay
        if input_types is None or not isinstance(input_types[0], tuple):
            raise RuntimeError("input_types argument should contain at least one tuple"
                               ", e.g. [((1, 14), np.float32)]")
        if isinstance(model, TabularNeuralNetTorchTvmPredictor):
            return model

        input_data = []
        for shape, dtype in input_types:
            if dtype == np.float32:
                input_data.append(torch.randn(shape, dtype=torch.float32))
            elif dtype == np.int32:
                input_data.append(torch.randint(low=0, high=1, size=(shape[0],)))
            else:
                raise NotImplementedError(f"not implemented dtype={dtype} for tvm compile.")
            
        scripted_model = torch.jit.trace(model, input_data).eval()

        shape_list = [(f"input{i}", t[0] if t[1] == np.float32 else (t[0][0],)) for i, t in enumerate(input_types)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        target = tvm.target.Target("llvm", host="llvm")
        dev = tvm.cpu(0)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

        model.architecture_desc['problem_type'] = model.problem_type
        TabularNeuralNetTorchTvmCompiler.save(lib, path, model.architecture_desc)

    @staticmethod
    def save(model, path: str, architecture_desc : dict = None) -> str:
        """Save the compiled model into tvm file format."""
        import json
        if architecture_desc is None:
            architecture_desc = {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path_lib = path + "model_tvm.tar"
        model.export_library(path_lib)
        with open(path + "model_architecture_desc.json", "wt") as fp:
            json.dump(architecture_desc, fp)
        return path_lib

    @staticmethod
    def load(path: str) -> TabularNeuralNetTorchTvmPredictor:
        """Load from the path that contains an tvm file."""
        import tvm, json
        path_lib = path + "model_tvm.tar"
        loaded_lib = tvm.runtime.load_module(path_lib)
        with open(path + "model_architecture_desc.json", "rt") as fp:
            architecture_desc = json.load(fp)
        return TabularNeuralNetTorchTvmPredictor(model=loaded_lib,
                                                 architecture_desc=architecture_desc)
