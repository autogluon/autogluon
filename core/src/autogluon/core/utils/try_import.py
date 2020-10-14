__all__ = ['try_import_mxboard']

def try_import_mxboard():
    try:
        import mxboard
    except ImportError:
        raise ImportError(
            "Unable to import dependency mxboard. "
            "A quick tip is to install via `pip install mxboard`. ")

