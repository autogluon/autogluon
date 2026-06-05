"""This is the autogluon version file."""
# NOTE(prototype): normally generated at build time by core/_setup_utils.create_version_file
# (and gitignored). With setup.py retired here, it's committed statically so the editable install
# imports. A full migration should generate it via a pyproject build hook, or drop it in favor of
# importlib.metadata (it's imported by __init__ and ~3 utils modules).

__version__ = "1.5.1"
__lite__ = False
