from pathlib import Path
import pytest

WHL_PATH = Path(__file__).parent.parent / "wheels"
WHL_PREFIX = "autogluon_lite"

import numpy as np
tests = [
    #
    # Regressions
    #
    {   # Default regression model on a small dataset
        'name': 'small regression',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : [ { 'predict' : {}, 'fit' : {} },          # All of the followiing params should return same results because they're defaults
                     { 'predict' : {}, 'fit' : { 'presets' : 'medium_quality_faster_train' } },
                     { 'predict' : {}, 'fit' : { 'presets' : 'ignore_text' } },
                     { 'predict' : {}, 'fit' : { 'hyperparameters' : 'default' } },
                     { 'predict' : { 'eval_metric' : 'root_mean_squared_error'}, 'fit' : { } },
                   ],
        'expected_score_range' : {
                  'CatBoost': (-7.86, 0.01),
                  'ExtraTreesMSE': (-7.88, 0.01),
                  'KNeighborsDist': (-8.69, 0.01),
                  'KNeighborsUnif': (-9.06, 0.01),
                  'LightGBM': (-15.55, 0.01),
                  'LightGBMLarge': (-10.43, 0.01),
                  'LightGBMXT': (-16.32, 0.01),
                  'NeuralNetFastAI': (-6.12, 0.01),
                  'NeuralNetTorch': (-4.96, 0.01),
                  'RandomForestMSE': (-9.63, 0.01),
                  'WeightedEnsemble_L2': (-5.66, 0.01),
                  'XGBoost': (-10.8, 0.01),
        },
    },
    {   # If we explicitly exclude some models the others should return unchanged and the ensemble result will be changed.
        'name': 'small regression excluded models',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : {}, 'fit' : { 'excluded_model_types' : [ 'KNN', 'RF', 'XT', 'GBM', 'CAT', 'XGB' ] } },
        'expected_score_range' : {
                  'NeuralNetFastAI': (-6.12, 0.01),
                  'NeuralNetTorch': (-4.96, 0.01),
                  'WeightedEnsemble_L2': (-5.09, 0.01),
        },
    },
    {   # Small regression, hyperparameters = light removes some models
        'name': 'small regression light hyperparameters',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : {}, 'fit' : { 'hyperparameters' : 'light' } },
        'expected_score_range' : {
                  'CatBoost': (-7.86, 0.01),
                  'ExtraTreesMSE': (-7.87, 0.01),
                  'LightGBM': (-15.55, 0.01),
                  'LightGBMLarge': (-10.43, 0.01),
                  'LightGBMXT': (-16.32, 0.01),
                  'NeuralNetFastAI': (-6.12, 0.01),
                  'NeuralNetTorch': (-4.96, 0.01),
                  'RandomForestMSE': (-9.63, 0.01),
                  'WeightedEnsemble_L2': (-5.66, 0.01),
                  'XGBoost': (-10.8, 0.01),
        },
    },
    {   # Small regression, hyperparameters = very_light removes some models
        'name': 'small regression very light hyperparameters',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : {}, 'fit' : { 'hyperparameters' : 'very_light' } },
        'expected_score_range' : {
                  'CatBoost': (-7.86, 0.01),
                  'LightGBM': (-15.55, 0.01),
                  'LightGBMLarge': (-10.43, 0.01),
                  'LightGBMXT': (-16.32, 0.01),
                  'NeuralNetFastAI': (-6.12, 0.01),
                  'NeuralNetTorch': (-4.96, 0.01),
                  'WeightedEnsemble_L2': (-5.58, 0.01),
                  'XGBoost': (-10.8, 0.01),
        },
    },
    {   # Small regression, hyperparameters = toy removes almost all models and runs very fast
        'name': 'small regression toy hyperparameters',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : {}, 'fit' : { 'hyperparameters' : 'toy' } },
        'expected_score_range' : {
                  'CatBoost': (-28.39, 0.01),
                  'LightGBM': (-27.81, 0.01),
                  'NeuralNetTorch': (-27.11, 0.01),
                  'WeightedEnsemble_L2': (-19.12, 0.01),
                  'XGBoost': (-19.12, 0.01),
        },
    },
    {   # High quality preset on small dataset.
        'name': 'small regression high quality',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : {}, 'fit' : { 'presets' : 'high_quality_fast_inference_only_refit' } },
        'expected_score_range' : {
                  'CatBoost_BAG_L1': (np.nan, np.nan),
                  'CatBoost_BAG_L1_FULL': (-7.75, 0.01),
                  'ExtraTreesMSE_BAG_L1': (-7.52, 0.01),
                  'ExtraTreesMSE_BAG_L1_FULL': (-7.52, 0.01),
                  'KNeighborsDist_BAG_L1': (-8.21, 0.01),
                  'KNeighborsDist_BAG_L1_FULL': (-8.21, 0.01),
                  'KNeighborsUnif_BAG_L1': (-8.7, 0.01),
                  'KNeighborsUnif_BAG_L1_FULL': (-8.7, 0.01),
                  'LightGBMLarge_BAG_L1': (np.nan, np.nan),
                  'LightGBMLarge_BAG_L1_FULL': (-9.94, 0.01),
                  'LightGBMXT_BAG_L1': (np.nan, np.nan),
                  'LightGBMXT_BAG_L1_FULL': (-13.03, 0.01),
                  'LightGBM_BAG_L1': (np.nan, np.nan),
                  'LightGBM_BAG_L1_FULL': (-14.17, 0.01),
                  'NeuralNetFastAI_BAG_L1': (np.nan, np.nan),
                  'NeuralNetFastAI_BAG_L1_FULL': (-5.48, 0.01),
                  'NeuralNetTorch_BAG_L1': (np.nan, np.nan),
                  'NeuralNetTorch_BAG_L1_FULL': (-5.29, 0.01),
                  'RandomForestMSE_BAG_L1': (-9.5, 0.01),
                  'RandomForestMSE_BAG_L1_FULL': (-9.5, 0.01),
                  'WeightedEnsemble_L2': (np.nan, np.nan),
                  'WeightedEnsemble_L2_FULL': (-5.29, 0.01),
                  'XGBoost_BAG_L1': (np.nan, np.nan),
                  'XGBoost_BAG_L1_FULL': (-9.76, 0.01),
        }
    },
    {   # Best quality preset on small dataset.
        'name': 'small regression best quality',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : {}, 'fit' : { 'presets' : 'best_quality' } },
        'expected_score_range' : {
                  'CatBoost_BAG_L1': (-7.85, 0.01),
                  'ExtraTreesMSE_BAG_L1' : (-7.52, 0.01),
                  'KNeighborsDist_BAG_L1' : (-8.21, 0.01),
                  'KNeighborsUnif_BAG_L1' : (-8.70, 0.01),
                  'LightGBMLarge_BAG_L1' : (-9.44, 0.01),
                  'LightGBMXT_BAG_L1' : (-14.78, 0.01),
                  'LightGBM_BAG_L1' : (-14.92, 0.01),
                  'NeuralNetFastAI_BAG_L1' : (-5.55, 0.01),
                  'NeuralNetTorch_BAG_L1' : (-5.07, 0.01),
                  'RandomForestMSE_BAG_L1' : (-9.5, 0.01),
                  'WeightedEnsemble_L2' : (-5.05, 0.01),   # beats default, as expected
                  'XGBoost_BAG_L1' : (-9.74, 0.01),
        }
    },
    {   # Default regression model, add some categorical features.
        'name': 'small regression with categorical',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 1,
        'dataset_hash' : '3e26d128e0',
        'params' : { 'predict' : {}, 'fit' : {} },          # Default params
        'expected_score_range' : {
                 'CatBoost': (-22.58, 0.01),
                 'ExtraTreesMSE': (-25.09, 0.01),
                 'KNeighborsDist': (-39.45, 0.01),
                 'KNeighborsUnif': (-35.64, 0.01),
                 'LightGBM': (-32.96, 0.01),
                 'LightGBMLarge': (-34.86, 0.01),
                 'LightGBMXT': (-32.69, 0.01),
                 'NeuralNetFastAI': (-22.11, 0.01),
                 'NeuralNetTorch': (-19.76, 0.01),
                 'RandomForestMSE': (-27.49, 0.01),
                 'WeightedEnsemble_L2': (-19.76, 0.01),
                 'XGBoost': (-24.93, 0.01),

        }
    },
    {   # Default regression model different metric
        'name': 'small regression metric mae',
        'type': 'regression',
        'n_samples': 100,
        'n_features': 2,
        'n_categorical': 0,
        'dataset_hash' : '5850a1c21a',
        'params' : { 'predict' : { 'eval_metric' : 'mean_absolute_error'}, 'fit' : { } },
        'expected_score_range' : {
                  'CatBoost': (-5.23, 0.01),
                  'ExtraTreesMSE': (-5.48, 0.01),
                  'KNeighborsDist': (-6.16, 0.01),
                  'KNeighborsUnif': (-6.61, 0.01),
                  'LightGBM': (-11.97, 0.01),
                  'LightGBMLarge': (-7.69, 0.01),
                  'LightGBMXT': (-12.37, 0.01),
                  'NeuralNetFastAI': (-4.74, 0.01),
                  'NeuralNetTorch': (-3.77, 0.01),
                  'RandomForestMSE': (-6.96, 0.01),
                  'WeightedEnsemble_L2': (-4.03, 0.01),
                  'XGBoost': (-8.32, 0.01),

        },
    },
    #
    # Classifications
    #
    {   # Default classification model on a small dataset
        'name': 'small classification',
        'type': 'classification',
        'n_samples': 400,  # With only 8 classes it's hard to compare model quality unless we have a biggest test set (and therefore train set).
        'n_features': 10,
        'n_informative': 5,
        'n_classes': 8,
        'n_categorical': 0,
        'dataset_hash' : 'be1f16df80',
        'params' : [ { 'predict' : {}, 'fit' : {} },     # All of the followiing params should return same results
                     { 'predict' : {}, 'fit' : { 'presets' : 'medium_quality_faster_train' } },
                     { 'predict' : {}, 'fit' : { 'presets' : 'ignore_text' } },
                     { 'predict' : {}, 'fit' : { 'hyperparameters' : 'default' } },
                     { 'predict' : { 'eval_metric' : 'accuracy'}, 'fit' : { } },
                   ],
        'expected_score_range' : {
                 'CatBoost': (0.245, 0.001),            # Classification scores are low numbers so we decrease the
                 'ExtraTreesEntr': (0.327, 0.001),      # tolerance to 0.001 to make sure we pick up changes.
                 'ExtraTreesGini': (0.32, 0.001),
                 'KNeighborsDist': (0.337, 0.001),
                 'KNeighborsUnif': (0.322, 0.001),
                 'LightGBM': (0.197, 0.001),
                 'LightGBMLarge': (0.265, 0.001),
                 'LightGBMXT': (0.23, 0.001),
                 'NeuralNetFastAI': (0.34, 0.001),
                 'NeuralNetTorch': (0.232, 0.001),
                 'RandomForestEntr': (0.305, 0.001),
                 'RandomForestGini': (0.295, 0.001),
                 'WeightedEnsemble_L2': (0.34, 0.001),
                 'XGBoost': (0.227, 0.001),
        }
    },
    {   # There's different logic for boolean classification so let's test that with n_classes = 2.
        'name': 'small classification boolean',
        'type': 'classification',
        'n_samples': 400,
        'n_features': 10,
        'n_informative': 5,
        'n_classes': 2,
        'n_categorical': 0,
        'dataset_hash' : '79e634aac3',
        'params' : [ { 'predict' : {}, 'fit' : {} },      # All of the followiing params should return same results
                     { 'predict' : { 'eval_metric' : 'accuracy'}, 'fit' : { } },
                   ],
        'expected_score_range' : {
                  'CatBoost': (0.61, 0.001),
                  'ExtraTreesEntr': (0.607, 0.001),
                  'ExtraTreesGini': (0.6, 0.001),
                  'KNeighborsDist': (0.61, 0.001),
                  'KNeighborsUnif': (0.61, 0.001),
                  'LightGBM': (0.632, 0.001),
                  'LightGBMLarge': (0.552, 0.001),
                  'LightGBMXT': (0.612, 0.001),
                  'NeuralNetFastAI': (0.62, 0.001),
                  'NeuralNetTorch': (0.597, 0.001),
                  'RandomForestEntr': (0.607, 0.001),
                  'RandomForestGini': (0.582, 0.001),
                  'WeightedEnsemble_L2': (0.61, 0.001),
                  'XGBoost': (0.58, 0.001),
        }
    },
]


@pytest.fixture
def selenium_standalone_micropip(selenium_standalone):
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

    from pytest_pyodide import spawn_web_server
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
@pytest.mark.lite
def test_load_data(selenium_standalone_micropip):
    from pytest_pyodide import run_in_pyodide
    @run_in_pyodide(packages=["xgboost", "pandas", "numpy", "scipy", "scikit-learn", "lightgbm"])
    async def run(selenium, test, train_data, test_data):
        expected_score_range = test['expected_score_range']
        expected_score_range = {k: (v[0], 0.02) for k, v in expected_score_range.items() if k not in [
            'CatBoost', 'NeuralNetFastAI', 'NeuralNetTorch',
        ]}

        from autogluon.tabular import TabularDataset, TabularPredictor

        # train_data = TabularDataset('./train-5k.csv')
        # subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
        # train_data = train_data.sample(n=subsample_size, random_state=0)
        print(train_data.head())
        label = 'label'
        print("Summary of class variable: \n", train_data[label].describe())
        save_path = 'agModels-predictClass'  # specifies folder to store trained models
        predictor = TabularPredictor(label=label, path=save_path).fit(train_data)

        # test_data = TabularDataset('./test-5k.csv')
        y_test = test_data[label]  # values to predict
        test_data_nolab = test_data.drop(columns=[label])  # delete label column to prove we're not cheating
        test_data_nolab.head()
        predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file

        y_pred = predictor.predict(test_data_nolab)
        print("Predictions:  \n", y_pred)
        perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
        leaderboard = predictor.leaderboard(test_data, silent=True)
        print(leaderboard[["model", "score_test", "score_val"]])
        trained_models = leaderboard[leaderboard.columns[0]]
        print(trained_models)
        assert set(trained_models) == set(expected_score_range.keys()), "Not all models were trained"
        for model in leaderboard['model']:
            score = leaderboard[leaderboard['model'] == model]['score_test'].values[0]
            score_range = expected_score_range[model]
            assert score_range[0] - score_range[1] <= score <= score_range[0] + score_range[1], f"Score for {model} ({score}) is out of expected range {score_range}"
            # TODO: check "score_val"

    test_case = tests[-1]
    from .utils import make_dataset
    DATA_TRAIN, DATA_TEST = make_dataset(request=test_case, seed=0)
    run(selenium_standalone_micropip, test_case, DATA_TRAIN, DATA_TEST)
