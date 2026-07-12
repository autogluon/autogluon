# AGENTS.md

> **Purpose.** This file and the `.agents/` directory give AI coding agents a
> machine-verified map of this repository — derived from AST analysis, not
> hand-written prose. It exists because AutoGluon's structure is *invisible*
> from the root layout and contains several non-obvious traps.
>
> **Methodology.** Progressive disclosure (Anthropic context-engineering
> guidance): this root file is a concise index; per-package detail lives in the
> machine-readable artifacts under `.agents/`.
>
> **For humans:** see [README.md](./README.md), [CONTRIBUTING.md](./CONTRIBUTING.md).

## What AutoGluon is

An AutoML toolkit (AWS) that trains high-accuracy ML/DL models on **tabular**,
**time-series**, and **multimodal** (image/text) data. Python 3.10–3.13, Apache-2.0.

## Critical fact #1 — this is a 7-member `uv` workspace monorepo

The root is **NOT** a single package. Six top-level directories are each
independently-publishable PEP 621 packages, plus a meta-package:

| Package dir | Imports as | Public entry point(s) | Stats |
|-------------|-----------|------------------------|-------|
| `common/`    | `autogluon.common`    | `Space` (HPO types) | [`packages.json`](./.agents/packages.json) |
| `core/`      | `autogluon.core`      | base `AbstractTrainer`/`Learner`, Ray HPO | [`packages.json`](./.agents/packages.json) |
| `features/`  | `autogluon.features`  | feature generators | [`packages.json`](./.agents/packages.json) |
| `multimodal/`| `autogluon.multimodal`| **`MultiModalPredictor`** | [`packages.json`](./.agents/packages.json) |
| `tabular/`   | `autogluon.tabular`   | **`TabularPredictor`** | [`packages.json`](./.agents/packages.json) |
| `timeseries/`| `autogluon.timeseries`| **`TimeSeriesPredictor`**, `TimeSeriesDataFrame` | [`packages.json`](./.agents/packages.json) |
| `autogluon/` | `autogluon`           | meta-package (depends on all above) | — |

**Layout convention:** source lives at `<pkg>/src/autogluon/<pkg>/`, tests at `<pkg>/tests/`.
Never edit code in the top-level `autogluon/` dir — it's packaging only.

## Critical fact #2 — the dependency graph has a cycle

Inter-package `import autogluon.X` edges, measured by AST (full data in
[`.agents/dependency_graph.json`](./.agents/dependency_graph.json)):

```
common ─┬─► core
        ├─► features
        └─► tabular, multimodal      ← ⚠ common imports the *upper* packages
core ───┼─► common, features, tabular
features► common
tabular ► common, core, features, multimodal
timeseries ► common, core, features, tabular
```

⚠️ **`common` imports `tabular`/`multimodal`** — a lower-layer package reaching
into upper layers. If you refactor or move code between packages, expect this
back-edge; don't "fix" it blindly (it may be type-only or lazy). Verify before
restructuring.

## Critical fact #3 — versions & dependency caps are dynamic

- Current version: see **`VERSION`** (e.g. `1.5.1`). `core/.../version.py` exposes `__version__`.
- Each package's `pyproject.toml` declares `version`/`dependencies` as **`dynamic`**.
- They are injected at build time by `<pkg>/setup.py`, which reads shared caps
  from **`core/_setup_utils.py`** — the single source of truth.
- **Do not hardcode versions or dependency bounds in a member's `pyproject.toml`.**
  Edit `core/_setup_utils.py` instead.
- Sibling deps stay local via root `[tool.uv.sources]`; never replace with PyPI pins.

## Build / install / test / lint

```bash
# Install (uv is the workspace manager; uv.lock is committed)
./full_install.sh            # one-shot, detects CPU/GPU
uv sync --all-extras         # resolve whole workspace from lock

# Editable install of one member:
pip install -e tabular/

# Tests — run per member from the member dir:
cd tabular   && pytest tests/ -x
cd timeseries && pytest tests/ -x
# (each member has its own conftest.py + 21–50 test files; see packages.json)

# Lint/format (Ruff, configured in root pyproject.toml):
ruff check . && ruff format .
pre-commit run --all-files
```
Ruff: `line-length=119`, `target-version=py310`, `isort.known-first-party=["autogluon"]`.

CI gate for contributors: `.github/workflows/continuous_integration.yml`.

## The three user-facing APIs (signatures extracted from source)

| Predictor | Location | `.fit()` key params | `.predict()` key params |
|-----------|----------|---------------------|-------------------------|
| `TabularPredictor` | `tabular/src/autogluon/tabular/predictor/predictor.py:91` | `train_data, tuning_data, time_limit, presets, hyperparameters` | `data, model, as_pandas, transform_features` |
| `TimeSeriesPredictor` | `timeseries/src/autogluon/timeseries/predictor.py:34` | `train_data, tuning_data, time_limit, presets, hyperparameters` | `data, known_covariates, model, use_cache` |
| `MultiModalPredictor` | `multimodal/src/autogluon/multimodal/predictor.py:36` | `train_data, presets, tuning_data, time_limit` | `data, candidate_data, id_mappings, as_pandas` |

Full method lists in [`api_surface.json`](./.agents/api_surface.json).

## Machine-readable assets (`.agents/`)

These are generated from AST/static analysis and are the most reliable source
for programmatic use:

- [`dependency_graph.json`](./.agents/dependency_graph.json) — inter-package import edges.
- [`packages.json`](./.agents/packages.json) — per-package stats (file/class/function counts, deps, test counts).
- [`api_surface.json`](./.agents/api_surface.json) — public classes + top-level functions per package.
- [`entry_points.json`](./.agents/entry_points.json) — Predictor/Space/data classes with `file:line` locations.
- [`code_chunks.json`](./.agents/code_chunks.json) — structure-aware (cAST-style) code chunks, each with a self-locating context header. For RAG / retrieval use.

## Common-task routing

| Task | Go to |
|------|-------|
| Add a tabular model | `tabular/src/autogluon/tabular/models/` + register in its `__init__.py` |
| Add a time-series model | `timeseries/src/autogluon/timeseries/models/` |
| Change default hyperparameters | `tabular/src/autogluon/tabular/configs/` |
| Modify public `TabularPredictor` API | `tabular/src/autogluon/tabular/predictor/predictor.py` |
| Bump version | `VERSION` (then `core/.../version.py` is regenerated) |
| Add docs (tutorials = .md, API refs = .rst/Sphinx) | `docs/` |

## Editing conventions

1. Namespace packages: imports are `autogluon.<pkg>.<sub>`. New code goes under `<pkg>/src/autogluon/<pkg>/`.
2. Don't hardcode versions/deps in member `pyproject.toml` — use `core/_setup_utils.py`.
3. Keep sibling deps workspace-local (root `[tool.uv.sources]`).
4. Match Ruff style (119 cols, isort first-party = `autogluon`).
5. One concern per PR (`.github/PULL_REQUEST_TEMPLATE.md`).

## Where to read more (humans)

[CONTRIBUTING.md](./CONTRIBUTING.md) · [AWESOME.md](./AWESOME.md) · [docs/index.md](./docs/index.md)
