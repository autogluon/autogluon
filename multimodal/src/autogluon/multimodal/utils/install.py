import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from ..constants import OBJECT_DETECTION, OCR

logger = logging.getLogger(__name__)


def check_if_packages_installed(problem_type: str = None, package_names: List[str] = None):
    """
    Check if necessary packages are installed for some problem types.
    Raise an error if an package can't be imported.

    Parameters
    ----------
    problem_type
        Problem type
    """
    if problem_type:
        problem_type = problem_type.lower()
        if any(p in problem_type for p in [OBJECT_DETECTION, OCR]):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import mmcv
            except ImportError as e:
                raise ValueError(
                    f"Encountered error while importing mmcv: {e}. {_get_mmlab_installation_guide('mmcv')}"
                )

            try:
                import mmdet
            except ImportError as e:
                raise ValueError(
                    f"Encountered error while importing mmdet: {e}. {_get_mmlab_installation_guide('mmdet')}"
                )

            if OCR in problem_type:
                try:
                    import mmocr
                except ImportError as e:
                    raise ValueError(
                        f'Encountered error while importing mmocr: {e}. Try to install mmocr: pip install "mmocr<1.0".'
                    )
    if package_names:
        for package_name in package_names:
            if package_name == "mmcv":
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        import mmcv
                    from mmcv import ConfigDict
                    from mmcv.runner import load_checkpoint
                    from mmcv.transforms import Compose
                except ImportError as e:
                    f"Encountered error while importing {package_name}: {e}. {_get_mmlab_installation_guide(package_name)}"
            elif package_name == "mmdet":
                try:
                    import mmdet
                    from mmdet.datasets.transforms import ImageToTensor
                    from mmdet.registry import MODELS
                except ImportError as e:
                    f"Encountered error while importing {package_name}: {e}. {_get_mmlab_installation_guide(package_name)}"
            elif package_name == "mmengine":
                try:
                    import mmengine
                    from mmengine.dataset import pseudo_collate as collate
                    from mmengine.runner import load_checkpoint
                except ImportError as e:
                    warnings.warn(e)
                    raise ValueError(
                        f"Encountered error while importing {package_name}: {e}. {_get_mmlab_installation_guide(package_name)}"
                    )
            else:
                raise ValueError(f"package_name {package_name} is not required.")


def _get_mmlab_installation_guide(package_name):
    if package_name == "mmdet":
        err_msg = 'Please install MMDetection by: pip install "mmdet==3.2.0"'
    elif package_name == "mmcv":
        err_msg = 'Please install MMCV by: mim install "mmcv==2.1.0"'
    elif package_name == "mmengine":
        err_msg = "Please install MMEngine by: mim install mmengine"
    else:
        raise ValueError("Available package_name are: mmdet, mmcv, mmengine.")

    err_msg += " Pytorch version larger than 2.1 is not supported yet. To use Autogluon for object detection, please downgrade PyTorch version to <=2.1."

    return err_msg
