import os
import numpy as np


class GraphModuleWrapper:
    """
    Wrap around GraphModule in tvm runtime, since it cannot be pickled.
    """
    def __init__(self, loaded_lib, dev):
        from tvm.contrib import graph_executor
        self.module = graph_executor.GraphModule(loaded_lib["default"](dev))

    def run(self, *args):
        return self.module.run(*args)

    def set_input(self, *args):
        return self.module.set_input(*args)

    def get_output(self, *args):
        return self.module.get_output(*args)

    def __getstate__(self):
        # No need to duplicate the model parameters here.
        return {}

    def __setstate__(self, values):
        pass


class TabularNeuralNetTorchTvmPredictor:
    def __init__(self, model, architecture_desc : dict = None):
        if architecture_desc is None:
            architecture_desc = {}
        import tvm
        dev = tvm.cpu()
        self.module = GraphModuleWrapper(model, dev)
        self.has_vector_features = architecture_desc.get("has_vector_features", False)
        self.has_embed_features = architecture_desc.get("has_embed_features", False)
        self.architecture_desc = architecture_desc

    def _get_input_vector(self, data_batch):
        input_data = []
        if self.has_vector_features:
            input_data.append(data_batch['vector'].to(self.device))
        if self.has_embed_features:
            embed_data = data_batch['embed']
            for i in range(len(self.architecture_desc['embed_dims'])):
                input_data.append(self.embed_blocks[i](embed_data[i].to(self.device)))

        if len(input_data) > 1:
            input_data = torch.cat(input_data, dim=1)
        else:
            input_data = input_data[0]
        return input_data

    def predict(self, X):
        """Run the model with the input and return the result."""
        input_data = self._get_input_vector(X)
        self.module.set_input("input0", input_data)
        self.module.run()
        out_shape = (batch_size, num_class)
        self.module.get_output(0, tvm.nd.empty(out_shape)).numpy()

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
                input_data.append(torch.randint(low=0, high=1, size=shape))
            else:
                raise NotImplementedError(f"not implemented dtype={dtype} for tvm compile.")
            
        scripted_model = torch.jit.trace(model, input_data).eval()

        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        target = tvm.target.Target("llvm", host="llvm")
        dev = tvm.cpu(0)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

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
        with open("model_architecture_desc.json", "wt") as fp:
            json.dump(architecture_desc, fp)
        return path_lib

    @staticmethod
    def load(path: str) -> TabularNeuralNetTorchTvmPredictor:
        """Load from the path that contains an tvm file."""
        import tvm, json
        path_lib = path + "model_tvm.tar"
        loaded_lib = tvm.runtime.load_module(path_lib)
        with open("model_architecture_desc.json", "rt") as fp:
            architecture_desc = json.load(fp)
        return TabularNeuralNetTorchTvmPredictor(model=loaded_lib,
                                                 architecture_desc=architecture_desc)
