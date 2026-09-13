# `best_quality_all_nf`

`best_quality_all_nf` extends the regular AutoGluon `best_quality` candidate pool with all 66 `NF...` aliases registered by the pinned `immanuelk1m/neuralforecast` bridge.

It keeps the same validation policy as `best_quality`:

- `num_val_windows="auto"`
- `refit_every_n_windows="auto"`
- weighted ensemble enabled unless the caller disables it

The model pool is `default` plus every model in `NEURALFORECAST_MODELS`. Existing presets are unchanged.

## Backend environment

The NeuralForecast fork must be installed in an interpreter that contains the dependencies required by the candidates you want to run. Point the preset at that interpreter with `AUTOGLUON_NF_PYTHON` before starting Python:

```bash
export AUTOGLUON_NF_PYTHON="$PWD/.venv-nf/bin/python"
```

If `AUTOGLUON_NF_PYTHON` is not set, the preset uses the current Python interpreter.

The bridge verifies that the backend is the pinned NeuralForecast fork revision. Optional research/foundation models can still require their own external checkpoints, source trees, or packages. AutoGluon will report model-specific failures; registering a candidate does not install those resources automatically.

## Usage

```python
from autogluon.timeseries import TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    prediction_length=4,
    freq="W-FRI",
    target="target",
    eval_metric="MAE",
)

predictor.fit(
    train_data,
    presets="best_quality_all_nf",
    time_limit=None,
)

print(predictor.leaderboard())
```

Use `time_limit=None` when the requirement is to give every candidate a chance to start. A finite global time limit can be exhausted before later candidates are reached.

## Models requiring additional task-specific inputs

The preset deliberately contains all 66 aliases, but not every architecture can be configured from a generic forecasting dataset alone. For example, `NFHINT` requires an explicit hierarchy (`S` and `hierarchy_item_ids`), while context/research/foundation adapters may require model-specific checkpoints or context configuration.

For such models, use the explicit `hyperparameters` interface when task-specific constructor arguments are required. Supplying `hyperparameters` directly overrides the preset's model pool, so build the desired combined dictionary from `get_hyperparameter_presets()["default_all_nf"]` before applying model-specific overrides.
