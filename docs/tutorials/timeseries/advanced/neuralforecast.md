# NeuralForecast fork models

This fork registers **all 66 public models** exported by
`immanuelk1m/neuralforecast` at commit
`0fd03a48191681864983f1fb2bb432db44ab3688`.
Each retains its original NeuralForecast implementation. The `NF` prefix keeps
these implementations distinct from AutoGluon's native models: `NFDeepAR` and
`DeepAR` are separate candidates.

Registration is not a claim that all pretrained checkpoints or research models
have been benchmarked. Some models require external source checkouts, trained
weights, compatible packages, or explicitly supplied contextual inputs.

## Installation

Install this AutoGluon checkout using its normal development installation. Create
an independent Python 3.11 environment for the NeuralForecast backend:

```bash
python3.11 -m venv .venv-nf
.venv-nf/bin/python -m pip install --upgrade pip
git clone https://github.com/immanuelk1m/neuralforecast.git external/neuralforecast
git -C external/neuralforecast checkout 0fd03a48191681864983f1fb2bb432db44ab3688
.venv-nf/bin/python -m pip install -e external/neuralforecast
```

Use an **editable checkout**: the pinned fork's packaging configuration lists only
the root package, so a regular wheel may omit model subpackages. Keep the checkout
at the pinned commit without modifying tracked files. The worker verifies both
revision and public exports before execution. Ordinary PyPI NeuralForecast has a
different model catalog and is deliberately rejected.

Install each optional backend's dependencies following the source fork's
`docs/exogenous_models.md`, `docs/exogenous_models_batch2.md`,
`docs/context_models.md`, and `docs/short_horizon_models.md` as appropriate.
Different candidates may use different `python_executable` environments. For
example, the existing Moirai adapter still needs its own `backend_python` for
uni2ts. No package installation or source cloning occurs inside fit/predict.

## Fit and compare

```python
from pathlib import Path
from autogluon.timeseries import TimeSeriesPredictor

# train_data: TimeSeriesDataFrame with target, observed inventory, and known holiday.
python = str(Path('.venv-nf/bin/python').absolute())
predictor = TimeSeriesPredictor(
    prediction_length=4,
    freq='W-FRI',
    target='target',
    eval_metric='MAE',
    known_covariates_names=['holiday'],
    quantile_levels=[0.1, 0.5, 0.9],
)
predictor.fit(
    train_data,
    hyperparameters={
        'Naive': {},
        'NFNHITS': {
            'python_executable': python,
            'input_size': 32,
            'max_steps': 300,
            'loss': 'MQLoss',
        },
        'NFMLP': {
            'python_executable': python,
            'input_size': 32,
            'max_steps': 300,
            'hidden_size': 128,
            'loss': 'MSE',
            'calibration_windows': 5,
        },
    },
    num_val_windows=3,
    refit_every_n_windows=2,
    time_limit=1800,
)
print(predictor.leaderboard())
# future_covariates must contain holiday for every item and forecast timestamp.
forecasts = predictor.predict(train_data, known_covariates=future_covariates)
```

Named classes are also importable, for example
`from autogluon.timeseries.models.neuralforecast import NFNHITSModel`.
`NeuralForecastModel` is the shared adapter and accepts `model_name='NHITS'`.
No existing preset is changed to run the 66 models automatically. Running every
optional model without its required resources would fail or trigger substantial
checkpoint downloads. Select prepared candidates explicitly.

Flat constructor hyperparameters, such as `learning_rate`, `input_size`, or
`hidden_size`, can be AutoGluon search spaces when HPO is requested. The worker
receives the resolved values. Nested NeuralForecast Auto* tuning wrappers are not
separate forecasting architectures in the source's public 66-model catalog.

## Bridge options

| Option | Meaning |
|---|---|
| `python_executable` | Backend Python; defaults to the current interpreter. Preserve venv symlinks with `absolute()`, not `resolve()`. |
| `model_name` | Required only to change the generic adapter's default NHITS; fixed for each `NF...` alias. |
| `validation_size` | Trailing observations **inside the AutoGluon training fold** for NF early stopping; default 0. Positive values must be at least the horizon. |
| `calibration_windows` | NF conformal windows for point-only models; default 2, increase for meaningful calibration. |
| `calibration_step_size` | Spacing between conformal origins; defaults to the horizon. |
| `predict_time_limit` | Optional hard deadline for each prediction subprocess, including imports/loading. |
| `hierarchy_item_ids` | Mandatory ordered item IDs for HINT; matches rows of S exactly. |

All other JSON-compatible values are forwarded to the original NF constructor.
`h` and `alias` are managed by the bridge. Losses use a string or a descriptor,
for example `{'name': 'DistributionLoss', 'kwargs': {'distribution': 'StudentT'}}`.
The same format works for `valid_loss`. Arbitrary Python objects/callbacks are not
transported across the process boundary. Use one device per model; configure
`accelerator` or the source adapter's `backend_device` explicitly for GPU use.

The default context is the next multiple of 32 covering `max(32, 2*horizon)`;
this is a convenience default, not model selection. Required patch lengths,
minimum contexts, supported horizons, and architecture-specific arguments remain
subject to the original model's validation. The bridge does not override
`max_steps`: inference-only source adapters retain their zero-step behavior.

## Covariates and panels

The adapter consults the source model's actual `EXOGENOUS_HIST`,
`EXOGENOUS_FUTR`, and `EXOGENOUS_STAT` flags. Numerical past, known-future and
static features map to `hist_exog_list`, `futr_exog_list`, and `stat_exog_list`.
Explicit lists can select subsets. Explicit unsupported lists fail. Implicitly
unused variables and categorical columns produce warnings and are recorded where
applicable; they are never promoted from past-only to known-future.

The bridge currently transports **numeric covariates**. Categorical embedding
APIs and arbitrary Python model objects are not transported. Encode categories
outside this bridge with a train-fitted transformation. Context-model text is
supplied through the source constructors' `contexts`/checkpoint settings and
numeric context IDs or precomputed text embedding columns. These retain the
source model's semantic requirements: inventory values are not text embeddings.

Histories must be finite. Missing targets are rejected explicitly; perform any
imputation within the relevant training boundaries. Prediction requires known
future values for the entire requested horizon. Target columns are prohibited
in the future frame. IDs are mapped to stable integer indices and outputs are
checked against exact item/timestamp keys, preventing accidental row-order swaps.

Joint multivariate models infer `n_series` and require a synchronized, complete
panel with the same items at fit and predict. Conformal calibration also requires
the calibrated item set. New items require refitting this adapter.

## Probabilistic outputs

Models with a native distribution or multi-quantile loss retain those outputs.
Configure `loss='MQLoss'` for supported quantile models; the quantile grid must
include all predictor quantiles. No missing quantiles are filled by interpolation
or copying a point prediction. Unsupported native grids, such as non-decile
TimesFM3 requests, retain the source adapter's explicit error.

Point-only models use NeuralForecast's own `PredictionIntervals` with
`conformal_distribution`. Calibration consumes **only the AutoGluon training
fold**; AutoGluon's outer validation frame is never passed into NF fitting or
calibration. This adds training/forecasting cost and requires sufficient history.
It does not give a distribution-free guarantee under arbitrary time-series
nonstationarity; assess coverage separately. Two windows are a minimal runnable
default, not a recommendation for reliable coverage.

AutoGluon requires a `mean` point-output column. The source's point output is used
when available; quantile-only outputs use P50 in that slot. P50 is a median, not an
estimated conditional mean. For mean-optimal metrics, choose a source model/loss
with the desired point statistic. Backend metadata records the quantile method
and original loss. Marginal quantiles alone are not joint sample paths.

## HINT

HINT needs an explicit hierarchy and a probabilistic base model:

```python
hyperparameters = {
    'NFHINT': {
        'python_executable': python,
        'base_model': 'NHITS',
        'hierarchy_item_ids': ['total', 'a', 'b'],
        'S': [[1, 1], [1, 0], [0, 1]],
        'reconciliation': 'BottomUp',
        'input_size': 32,
        'max_steps': 300,
    },
}
```

All hierarchy rows must be present on the same timestamp grid. The bottom-level
identity block must be last. The bridge preserves S and reconciliation alongside
the NF base checkpoint and reconstructs the original HINT wrapper on reload,
because the source HINT `save()` persists only its base network. Coherent sample
means do not imply that summing marginal quantiles produces coherent quantiles.

## Persistence, resources, and tests

Fit/predict run in isolated subprocesses and exchange private CSV/JSON files. The
fit deadline covers imports, calibration, training, and checkpoint writing.
On POSIX a timeout terminates the process group, including nested interpreters.
Windows terminates the immediate worker; nested backend process cleanup has not
been validated there. Prediction deadlines are separately configurable.

Native NF checkpoints and calibration metadata live below the AutoGluon model
directory. AutoGluon save/load, refit-full copies, and directory relocation retain
them. External checkpoints, source directories, and backend interpreter paths
remain external dependencies and must exist in the destination environment.
Checkpoint deserialization requires **trusted files**, just like the underlying
PyTorch/AutoGluon loaders. No external model weights or research source are
redistributed. Every source/checkpoint's license still applies; in particular,
the source fork documents separate non-commercial restrictions on TimesFM3
weights. Review those terms before deployment.

`timeseries/tests/neuralforecast_bridge/test_contract.py` tests protocol boundaries
without NF or AutoGluon imports. `test_integration.py` uses real CPU models when
`NF_PYTHON` is set: MLP, NHITS, DeepAR, iTransformer, HINT, point conformal
calibration, covariates, serialization/relocation, multi-window validation,
ensembling, refit-full, and timeout handling. It also compares all 66 aliases
against the actual source exports. Pretrained checkpoint inference, every
research architecture, GPU execution, and benchmark accuracy are **not covered**
by these CPU smoke tests. Check the PR's actual CI result before relying on them.

```bash
NF_PYTHON=/absolute/path/to/.venv-nf/bin/python \
python -m pytest -q -o addopts='' \
  --confcutdir=timeseries/tests/neuralforecast_bridge \
  timeseries/tests/neuralforecast_bridge
```

## Complete model catalog

| NeuralForecast model | AutoGluon alias |
|---|---|
| `RNN` | `NFRNN` |
| `GRU` | `NFGRU` |
| `LSTM` | `NFLSTM` |
| `TCN` | `NFTCN` |
| `DeepAR` | `NFDeepAR` |
| `DilatedRNN` | `NFDilatedRNN` |
| `MLP` | `NFMLP` |
| `NHITS` | `NFNHITS` |
| `NBEATS` | `NFNBEATS` |
| `NBEATSx` | `NFNBEATSx` |
| `DLinear` | `NFDLinear` |
| `NLinear` | `NFNLinear` |
| `TFT` | `NFTFT` |
| `VanillaTransformer` | `NFVanillaTransformer` |
| `Informer` | `NFInformer` |
| `Autoformer` | `NFAutoformer` |
| `PatchTST` | `NFPatchTST` |
| `FEDformer` | `NFFEDformer` |
| `StemGNN` | `NFStemGNN` |
| `HINT` | `NFHINT` |
| `TimesNet` | `NFTimesNet` |
| `TimeLLM` | `NFTimeLLM` |
| `TSMixer` | `NFTSMixer` |
| `TSMixerx` | `NFTSMixerx` |
| `MLPMultivariate` | `NFMLPMultivariate` |
| `iTransformer` | `NFiTransformer` |
| `BiTCN` | `NFBiTCN` |
| `TiDE` | `NFTiDE` |
| `DeepNPTS` | `NFDeepNPTS` |
| `SOFTS` | `NFSOFTS` |
| `SOFTSSharp` | `NFSOFTSSharp` |
| `TimeMixer` | `NFTimeMixer` |
| `KAN` | `NFKAN` |
| `RMoK` | `NFRMoK` |
| `TimeXer` | `NFTimeXer` |
| `xLSTM` | `NFxLSTM` |
| `XLinear` | `NFXLinear` |
| `VoT` | `NFVoT` |
| `GPT4MTS` | `NFGPT4MTS` |
| `UniTime` | `NFUniTime` |
| `LangTime` | `NFLangTime` |
| `Aurora` | `NFAurora` |
| `ChatTime` | `NFChatTime` |
| `TabPFNTS` | `NFTabPFNTS` |
| `CrossLinear` | `NFCrossLinear` |
| `TimerXL` | `NFTimerXL` |
| `TinyTimeMixer` | `NFTinyTimeMixer` |
| `Chronos2` | `NFChronos2` |
| `Moirai` | `NFMoirai` |
| `MoiraiMoE` | `NFMoiraiMoE` |
| `TimesFM` | `NFTimesFM` |
| `Toto` | `NFToto` |
| `DAG` | `NFDAG` |
| `KITE` | `NFKITE` |
| `GLAFF` | `NFGLAFF` |
| `APT` | `NFAPT` |
| `Moirai2` | `NFMoirai2` |
| `ChronosX` | `NFChronosX` |
| `BaguanTS` | `NFBaguanTS` |
| `RAG4CTS` | `NFRAG4CTS` |
| `SpecTF` | `NFSpecTF` |
| `TGForecaster` | `NFTGForecaster` |
| `TimesFM3` | `NFTimesFM3` |
| `SeesawNet` | `NFSeesawNet` |
| `Dualformer` | `NFDualformer` |
| `SearchCast` | `NFSearchCast` |
