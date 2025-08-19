"""Test autogluon.tabular on pyodide.
The test needs to be run via pyodide/tools/pytest_wrapper.py,
and using conftest.py from pyodide root directory.
"""

from pathlib import Path

import pytest

ROOT_PATH = Path(__file__).parent.parent.parent.parent
WHL_PREFIX = "autogluon_lite"


@pytest.fixture
def selenium_standalone_micropip(selenium_standalone):
    wheel_paths = []
    for regex_path_str in [
        ("common", f"{WHL_PREFIX}.common-*.whl"),
        ("core", f"{WHL_PREFIX}.core-*.whl"),
        ("features", f"{WHL_PREFIX}.features-*.whl"),
        ("tabular", f"{WHL_PREFIX}.tabular-*.whl"),
        ("autogluon", f"{WHL_PREFIX}-*.whl"),
    ]:
        wheel_path = [
            f"{regex_path_str[0]}/dist/{w.name}"
            for w in (ROOT_PATH / regex_path_str[0] / "dist").glob(regex_path_str[1])
        ]
        assert len(wheel_path) == 1
        wheel_path = wheel_path[0]
        wheel_paths.append(wheel_path)

    from pytest_pyodide import spawn_web_server

    with spawn_web_server(ROOT_PATH) as server:
        server_hostname, server_port, _ = server
        base_url = f"http://{server_hostname}:{server_port}/"
        selenium_standalone.run_js(
            f"""
            await pyodide.loadPackage("micropip");
            pyodide.runPython("import micropip");
            await pyodide.runPythonAsync(`
                import micropip
                await micropip.install('{base_url + wheel_paths[0]}')
                await micropip.install('{base_url + wheel_paths[1]}')
                await micropip.install('{base_url + wheel_paths[2]}')
                await micropip.install('{base_url + wheel_paths[3]}')
                await micropip.install('{base_url + wheel_paths[4]}')
            `);
            """
        )
    yield selenium_standalone


@pytest.mark.skip_refcount_check
@pytest.mark.driver_timeout(60)
@pytest.mark.pyodide
def test_train_classifier(selenium_standalone_micropip):
    from pytest_pyodide import run_in_pyodide

    @run_in_pyodide(packages=["xgboost", "pandas", "numpy", "scipy", "scikit-learn", "lightgbm"])
    async def run(selenium, test, train_data, test_data):
        expected_score_range = test["expected_score_range"]
        expected_score_range = {
            k: (v[0], 0.02)
            for k, v in expected_score_range.items()
            if k
            not in [
                "CatBoost",
                "NeuralNetFastAI",
                "NeuralNetTorch",
            ]
        }

        from autogluon.tabular import TabularPredictor

        print(train_data.head())
        label = "label"
        print("Summary of class variable: \n", train_data[label].describe())
        save_path = "agModels-predictClass"  # specifies folder to store trained models
        predictor = TabularPredictor(label=label, path=save_path).fit(train_data)

        y_test = test_data[label]  # values to predict
        test_data_nolab = test_data.drop(columns=[label])  # delete label column to prove we're not cheating
        test_data_nolab.head()
        predictor = TabularPredictor.load(save_path)

        y_pred = predictor.predict(test_data_nolab)
        print("Predictions:  \n", y_pred)
        perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
        leaderboard = predictor.leaderboard(test_data)
        print(leaderboard[["model", "score_test", "score_val"]])
        trained_models = leaderboard[leaderboard.columns[0]]
        print(trained_models)
        assert set(trained_models) == set(expected_score_range.keys()), "Not all models were trained"
        for model in leaderboard["model"]:
            score = leaderboard[leaderboard["model"] == model]["score_test"].values[0]
            score_range = expected_score_range[model]
            assert score_range[0] - score_range[1] <= score <= score_range[0] + score_range[1], (
                f"Score for {model} ({score}) is out of expected range {score_range}"
            )

    from .utils import make_dataset, tests

    test_case = tests[-1]
    data_train, data_test = make_dataset(request=test_case, seed=0)
    run(selenium_standalone_micropip, test_case, data_train, data_test)
