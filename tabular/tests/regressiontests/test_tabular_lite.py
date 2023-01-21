from pathlib import Path
import pytest
from pytest_pyodide import run_in_pyodide, spawn_web_server

DEMO_PATH = Path(__file__).parent / "test_data"
WHL_PATH = Path(__file__).parent / "wheels"

WHL_PREFIX = "autogluon-lite"

AG_WHL_NAME = [w.name for w in WHL_PATH.glob(f"{WHL_PREFIX}-*-py3-none-any.whl")]
assert len(AG_WHL_NAME) == 1
AG_WHL_NAME = AG_WHL_NAME[0]

AG_COMMON_WHL_NAME = [w.name for w in WHL_PATH.glob(f"{WHL_PREFIX}.common-*-py3-none-any.whl")]
assert len(AG_COMMON_WHL_NAME) == 1
AG_COMMON_WHL_NAME = AG_COMMON_WHL_NAME[0]

AG_CORE_WHL_NAME = [w.name for w in WHL_PATH.glob(f"{WHL_PREFIX}.core-*-py3-none-any.whl")]
assert len(AG_CORE_WHL_NAME) == 1
AG_CORE_WHL_NAME = AG_CORE_WHL_NAME[0]

AG_FEATURE_WHL_NAME = [w.name for w in WHL_PATH.glob(f"{WHL_PREFIX}.features-*-py3-none-any.whl")]
assert len(AG_FEATURE_WHL_NAME) == 1
AG_FEATURE_WHL_NAME = AG_FEATURE_WHL_NAME[0]

AG_TAB_WHL_NAME = [w.name for w in WHL_PATH.glob(f"{WHL_PREFIX}.tabular-*-py3-none-any.whl")]
assert len(AG_TAB_WHL_NAME) == 1
AG_TAB_WHL_NAME = AG_TAB_WHL_NAME[0]


@pytest.fixture
def selenium_standalone_micropip(selenium_standalone):
    with spawn_web_server(WHL_PATH) as server:
        server_hostname, server_port, _ = server
        base_url = f"http://{server_hostname}:{server_port}/"
        url_ag = base_url + AG_WHL_NAME
        url_ag_common = base_url + AG_COMMON_WHL_NAME
        url_ag_core = base_url + AG_CORE_WHL_NAME
        url_ag_features = base_url + AG_FEATURE_WHL_NAME
        url_ag_tab = base_url + AG_TAB_WHL_NAME
        selenium_standalone.run_js(
            f"""
            await pyodide.loadPackage("micropip");
            pyodide.runPython("import micropip");
            await pyodide.runPythonAsync(`
                import micropip
                await micropip.install('{url_ag_common}')
                await micropip.install('{url_ag_core}')
                await micropip.install('{url_ag_features}')
                await micropip.install('{url_ag_tab}')
                await micropip.install('{url_ag}')
            `);
            """
        )
    yield selenium_standalone


@pytest.mark.skip_refcount_check
@pytest.mark.driver_timeout(60)
def test_train(selenium_standalone_micropip):
    @run_in_pyodide(packages=["xgboost", "pandas", "numpy", "scipy", "scikit-learn", "lightgbm"])
    async def run(selenium, data_train, data_test):
        # TODO: replace with `(dftrain, dftest) = make_dataset(request=test, seed=0)`
        with open("train-5k.csv", "wb") as f:
            f.write(data_train)
        with open("test-5k.csv", "wb") as f:
            f.write(data_test)

        expected_score_range = {
            'ExtraTreesEntr': (0.822, 0.001),
            'ExtraTreesGini': (0.82, 0.001),
            'KNeighborsDist': (0.726, 0.001),
            'KNeighborsUnif': (0.763, 0.001),
            'LightGBM': (0.836, 0.001),
            'LightGBMLarge': (0.827, 0.001),
            'LightGBMXT': (0.833, 0.001),
            'RandomForestEntr': (0.832, 0.001),
            'RandomForestGini': (0.83, 0.001),
            'WeightedEnsemble_L2': (0.831, 0.001),
            'XGBoost': (0.831, 0.001),
        }

        from autogluon.tabular import TabularDataset, TabularPredictor
        train_data = TabularDataset('./train-5k.csv')
        subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
        train_data = train_data.sample(n=subsample_size, random_state=0)
        print(train_data.head())
        label = 'class'
        print("Summary of class variable: \n", train_data[label].describe())
        save_path = 'agModels-predictClass'  # specifies folder to store trained models
        predictor = TabularPredictor(label=label, path=save_path).fit(train_data)

        test_data = TabularDataset('./test-5k.csv')
        y_test = test_data[label]  # values to predict
        test_data_nolab = test_data.drop(columns=[label])  # delete label column to prove we're not cheating
        test_data_nolab.head()
        predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file

        y_pred = predictor.predict(test_data_nolab)
        print("Predictions:  \n", y_pred)
        perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
        leaderboard = predictor.leaderboard(test_data, silent=True)
        trained_models = leaderboard[leaderboard.columns[0]]
        print(trained_models)
        assert set(trained_models) == set(expected_score_range.keys()), "Not all models were trained"
        for model in leaderboard['model']:
            score = leaderboard[leaderboard['model'] == model]['score_test'].values[0]
            score_range = expected_score_range[model]
            assert score_range[0] <= score <= score_range[0] + score_range[1], f"Score for {model} ({score}) is out of expected range {score_range}"
            # TODO: check "score_val"

    DATA_TRAIN = (DEMO_PATH / "train-5k.csv").read_bytes()
    DATA_TEST = (DEMO_PATH / "test-5k.csv").read_bytes()
    run(selenium_standalone_micropip, DATA_TRAIN, DATA_TEST)