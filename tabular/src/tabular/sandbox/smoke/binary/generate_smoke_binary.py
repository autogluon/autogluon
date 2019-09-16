
from f3_grail_data_frame_utilities.loaders import load_pd
from f3_grail_data_frame_utilities.savers import save_pd


# Get input data here: https://drive.corp.amazon.com/folders/Grail/AutoML/data/sandbox/mla_nlp1
if __name__ == '__main__':
    path_prefix = 'data/sandbox/mla_nlp1/'
    path_train = path_prefix + 'inputs/training.csv'
    path_test = path_prefix + 'inputs/public_test_features.csv'

    X_train = load_pd.load(path=path_train, encoding='latin1', dtype=str)
    X_test = load_pd.load(path=path_test, encoding='latin1', dtype=str)

    X_train_smoke = X_train.sample(100, random_state=0).reset_index(drop=True)
    X_test_smoke = X_test.sample(100, random_state=0).reset_index(drop=True)

    path_prefix_smoke = 'test/data/sandbox/smoke/binary/'
    path_train_smoke = path_prefix_smoke + 'inputs/training.csv'
    path_test_smoke = path_prefix_smoke + 'inputs/test.csv'

    save_pd.save(path=path_train_smoke, df=X_train_smoke)
    save_pd.save(path=path_test_smoke, df=X_test_smoke)
