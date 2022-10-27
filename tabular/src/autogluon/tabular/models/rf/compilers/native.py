import os
import pickle


class RFNativeCompiler:
    name = 'native'
    save_in_pkl = True

    @staticmethod
    def can_compile():
        return True

    @staticmethod
    def compile(obj, path: str):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(obj.model, (RandomForestClassifier, RandomForestRegressor,
                                  ExtraTreesClassifier, ExtraTreesRegressor)):
            with open(path + 'model_native.pkl', 'wb') as fp:
                fp.write(pickle.dumps(obj.model))
            return obj.model
        else:
            return RFNativeCompiler.load(obj=obj, path=path)

    @staticmethod
    def load(obj, path: str):
        pkl = None
        with open(path + 'model_native.pkl', 'rb') as fp:
            pkl = fp.read()
        return pickle.loads(pkl)
