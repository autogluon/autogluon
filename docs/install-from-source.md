# Installing AutoGluon from source

This is the workflow for installing AutoGluon locally from a git clone — for development, or to
run the latest unreleased code. AutoGluon is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/),
so the entire monorepo installs editable in **one command**.

## Prerequisites

- **Python 3.10–3.13**
- **git**
- **[uv](https://docs.astral.sh/uv/)** — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh   # or: pip install -U uv
  ```

## Quickstart (recommended)

```bash
git clone https://github.com/autogluon/autogluon
cd autogluon
uv sync --all-extras
```

`uv sync` creates a `.venv/` at the repo root and installs **all 7 `autogluon.*` packages editable**
(`common`, `core`, `features`, `tabular`, `timeseries`, `multimodal`, and the meta `autogluon`),
resolved deterministically from the committed `uv.lock`.

To reproduce the exact locked environment (fail instead of re-resolving if the lock is out of date):
```bash
uv sync --frozen --all-extras
```

Run code in the environment without activating it:
```bash
uv run python -c "from autogluon.tabular import TabularPredictor; print('ok')"
```
…or activate it:
```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**Extras:**
- `uv sync` (no flags) → base dependencies only (skeleton install, no optional models).
- `uv sync --all-extras` → the full optional set across every package (equivalent to
  `pip install autogluon` plus all submodule extras).

## CPU-only (no CUDA)

The default resolution pulls the standard PyTorch build (CUDA on Linux). For a CPU-only machine,
point at the PyTorch CPU index:
```bash
uv sync --all-extras --index https://download.pytorch.org/whl/cpu
```
(On macOS, PyTorch is CPU-only by default, so `uv sync --all-extras` is enough.)

## Installing a single submodule

Sync just one package with chosen extras — sibling `autogluon.*` packages resolve automatically
from the workspace:
```bash
# tabular with only LightGBM + CatBoost
uv sync --package autogluon.tabular --extra lightgbm --extra catboost

# tabular with the full model set  (== pip install "autogluon.tabular[all]")
uv sync --package autogluon.tabular --extra all
```
Tabular extras in `all`: `lightgbm, catboost, xgboost, fastai, tabm, mitra, ray`.
Additional extras: `tabicl, tabpfn, tabdpt, realmlp, interpret, imodels, skex, skl2onnx, tabpfnmix`,
and the benchmark bundle `tabarena`.

## Installing directly from GitHub (no clone)

You can install a package straight from the repository using a
[PEP 508 direct reference](https://peps.python.org/pep-0508/#examples) with `#subdirectory=`
pointing at the submodule's folder — handy for a one-off install of an unreleased version without
cloning the workspace:

```bash
# meta package (full stack), latest master
pip install "autogluon @ git+https://github.com/autogluon/autogluon.git#subdirectory=autogluon"

# a single submodule, with extras
pip install "autogluon.tabular[all] @ git+https://github.com/autogluon/autogluon.git#subdirectory=tabular"

# pin a branch, tag, or commit with @<ref>
pip install "autogluon.tabular @ git+https://github.com/autogluon/autogluon.git@master#subdirectory=tabular"
```
(`uv pip install "<same spec>"` works too.)

> **Note:** this builds only the requested subdirectory, so its sibling `autogluon.*` dependencies
> (e.g. `autogluon.core`, `autogluon.features`) are resolved from **PyPI** at the exact pinned
> build version — not from git. It therefore works cleanly when that version is already published.
> To get *every* `autogluon.*` package from the same git source, use the `uv sync` clone flow above
> (its workspace sources resolve the siblings locally).

## Development workflow

```bash
uv run pytest tabular/tests          # run a package's tests
uv run pre-commit install            # one-time: enable formatting hooks
uv run pre-commit run --all-files    # lint/format like CI (ruff)
```

Pick a specific interpreter with `uv sync -p 3.12` (or `uv python pin 3.12`).

## Keeping in sync

`uv.lock` is committed, so installs are reproducible. After pulling changes, re-run
`uv sync --all-extras`. If you **change a package's dependencies** (e.g. a cap in
`core/_setup_utils.py`), refresh and commit the lockfile — CI runs `uv lock --check` and fails if
it's stale:
```bash
uv lock
```
Resolution is pinned to a fixed date via `exclude-newer` in the root `pyproject.toml`, so a newly
released dependency won't change the lock (or turn CI red) until you bump that date alongside the
`uv lock` refresh.

## Alternative: `uv pip install` / convenience script

`uv pip install` also honors the workspace, so it builds a package and resolves its
`autogluon.*` siblings from the local source automatically — no need to list them:
```bash
uv pip install "./autogluon"        # full stack
uv pip install "./tabular[all]"     # just tabular + its stack
```
For an editable install of every package in one go, `./full_install.sh` also works.
