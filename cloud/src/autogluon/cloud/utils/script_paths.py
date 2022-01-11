import os

from pathlib import Path

CLOUD_PATH = Path(__file__).parent.parent.absolute()
SCRIPTS_PATH = os.path.join(CLOUD_PATH, 'scripts')
TRAIN_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, 'train.py')
TABULAR_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, 'tabular_serve.py')
TEXT_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, 'text_serve.py')
