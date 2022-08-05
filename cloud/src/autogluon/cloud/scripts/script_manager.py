import os

from distutils.version import LooseVersion
from pathlib import Path

CLOUD_PATH = Path(__file__).parent.parent.absolute()
SCRIPTS_PATH = os.path.join(CLOUD_PATH, 'scripts')
TRAIN_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, 'train.py')
TABULAR_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, 'tabular_serve.py')
TEXT_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, 'text_serve.py')


class ScriptManager:

    CLOUD_PATH = Path(__file__).parent.parent.absolute()
    SCRIPTS_PATH = os.path.join(CLOUD_PATH, 'scripts')
    TRAIN_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, 'train.py')
    TABULAR_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, 'tabular_serve.py')
    TEXT_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, 'text_serve.py')
    TEXT_AUTOMM_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, 'text_serve_automm.py')
    IMAGE_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, 'image_serve.py')
    _SERVE_SCRIPT_MAP = dict(
        tabular=TABULAR_SERVE_SCRIPT_PATH,
        text=TEXT_SERVE_SCRIPT_PATH,
        text_automm=TEXT_AUTOMM_SERVE_SCRIPT_PATH,
        image=IMAGE_SERVE_SCRIPT_PATH,
    )

    @classmethod
    def get_train_script(cls, predictor_type, framework_version):
        assert predictor_type in ['tabular', 'text', 'image']
        # tabular, text and image share the same training script
        return TRAIN_SCRIPT_PATH

    @classmethod
    def get_serve_script(cls, predictor_type, framework_version):
        assert predictor_type in ['tabular', 'text', 'image']
        if predictor_type == 'text' and LooseVersion(framework_version) >= LooseVersion('0.4'):
            predictor_type = 'text_automm'
        return cls._SERVE_SCRIPT_MAP[predictor_type]
