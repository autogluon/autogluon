
from f3_grail_data_frame_utilities.loaders import load_pd
from f3_grail_data_frame_utilities.savers import save_pd
from tabular.ml.learner.default_learner import DefaultLearner


# data can be string path to data or the loaded data in DataFrame format
# learner can be string path to learner or the loaded learner object
# save_path [optional] will save y_pred to the indicated location
def predict(data, learner, save_path=None, save_path_proba=None, reset_paths=False):
    if type(data) == str:
        data = load_pd.load(data, encoding='latin1')
    if type(learner) == str:
        learner = DefaultLearner.load(learner, reset_paths=reset_paths)

    y_pred = learner.predict_and_submit(data, save=False)

    if save_path is not None:
        save_pd.save(path=save_path, df=y_pred)

    return y_pred
