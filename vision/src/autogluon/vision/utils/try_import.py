__all__ = ['try_import_cv2']


def try_import_cv2():
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "Unable to import dependency cv2. "
            "A quick tip is to install via `pip install opencv-python`. ")

