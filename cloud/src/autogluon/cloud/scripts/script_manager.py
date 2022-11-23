import os
from pathlib import Path

CLOUD_PATH = Path(__file__).parent.parent.absolute()
SCRIPTS_PATH = os.path.join(CLOUD_PATH, "scripts")
TRAIN_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, "train.py")
TABULAR_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, "tabular_serve.py")
TEXT_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, "text_serve.py")


class ScriptManager:

    CLOUD_PATH = Path(__file__).parent.parent.absolute()
    SCRIPTS_PATH = os.path.join(CLOUD_PATH, "scripts")
    TRAIN_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, "train.py")
    TABULAR_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, "tabular_serve.py")
    TEXT_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, "text_serve.py")
    IMAGE_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, "image_serve.py")
    MULTIMODAL_SERVE_SCRIPT_PATH = os.path.join(SCRIPTS_PATH, "multimodal_serve.py")
    _SERVE_SCRIPT_MAP = dict(
        tabular=TABULAR_SERVE_SCRIPT_PATH,
        text=TEXT_SERVE_SCRIPT_PATH,
        image=IMAGE_SERVE_SCRIPT_PATH,
        multimodal=MULTIMODAL_SERVE_SCRIPT_PATH,
    )

    @classmethod
    def get_train_script(cls, predictor_type, framework_version):
        assert predictor_type in ["tabular", "text", "image", "multimodal"]
        # tabular, text and image share the same training script
        return TRAIN_SCRIPT_PATH

    @classmethod
    def get_serve_script(cls, predictor_type, framework_version):
        assert predictor_type in ["tabular", "text", "image", "multimodal"]
        return cls._SERVE_SCRIPT_MAP[predictor_type]
