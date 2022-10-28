import os
import pickle


class AbstractNativeCompiler:
    name = 'native'
    save_in_pkl = True

    @staticmethod
    def can_compile():
        return True

    @staticmethod
    def compile(model, input_types=None):
        """
        Compile the trained model for faster inference.

        Parameters
        ----------
        model
            The native model that is expected to be compiled.
        input_types : list, default=None
            A list of tuples containing shape and element type info, e.g. [((1, 14), np.float32),].
            The list would be used as the input data for the model.
            The compiler would optimize the model to perform best with the given input type.

        Returns
        -------
        model
            The optimized predictor with predict() and predict_proba() interface.
        """
        return model

    @staticmethod
    def save(model, path: str):
        with open(path + 'model_native.pkl', 'wb') as fp:
            fp.write(pickle.dumps(model))

    @staticmethod
    def load(path: str):
        pkl = None
        with open(path + 'model_native.pkl', 'rb') as fp:
            pkl = fp.read()
        return pickle.loads(pkl)


RFNativeCompiler = AbstractNativeCompiler
