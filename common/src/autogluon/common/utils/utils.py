from datetime import datetime

import logging
import os

logger = logging.getLogger(__name__)


def setup_outputdir(path, warn_if_exist=True, create_dir=True, path_suffix=None):
    if path_suffix is None:
        path_suffix = ''
    if path_suffix and path_suffix[-1] == os.path.sep:
        path_suffix = path_suffix[:-1]
    if path is not None:
        path = f'{path}{path_suffix}'
    if path is None:
        utcnow = datetime.utcnow()
        timestamp = utcnow.strftime("%Y%m%d_%H%M%S")
        path = f"AutogluonModels/ag-{timestamp}{path_suffix}{os.path.sep}"
        for i in range(1, 1000):
            try:
                if create_dir:
                    os.makedirs(path, exist_ok=False)
                    break
                else:
                    if os.path.isdir(path):
                        raise FileExistsError
                    break
            except FileExistsError as e:
                path = f"AutogluonModels/ag-{timestamp}-{i:03d}{path_suffix}{os.path.sep}"
        else:
            raise RuntimeError("more than 1000 jobs launched in the same second")
        logger.log(25, f'No path specified. Models will be saved in: "{path}"')
    elif warn_if_exist:
        try:
            if create_dir:
                os.makedirs(path, exist_ok=False)
            elif os.path.isdir(path):
                raise FileExistsError
        except FileExistsError as e:
            logger.warning(f'Warning: path already exists! This predictor may overwrite an existing predictor! path="{path}"')
    path = os.path.expanduser(path)  # replace ~ with absolute path if it exists
    if path[-1] != os.path.sep:
        path = path + os.path.sep
    return path
