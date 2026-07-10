REM Install all AutoGluon packages from source via the uv workspace (creates .venv/).
REM See docs/install-from-source.md for details and options.
python -m pip install -U uv
python -m uv sync --all-extras
