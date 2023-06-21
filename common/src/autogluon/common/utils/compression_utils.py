def _gzip_open(*args, **kwargs):
    import gzip

    return gzip.open(*args, **kwargs)


def _bz2_open(*args, **kwargs):
    import bz2

    return bz2.open(*args, **kwargs)


# Lazy import to avoid the following issue if users installed via pyenv:
#  https://stackoverflow.com/questions/57743230/userwarning-could-not-import-the-lzma-module-your-installed-python-is-incomple
def _lzma_open(*args, **kwargs):
    import lzma

    return lzma.open(*args, **kwargs)


compression_fn_map = {
    None: {
        "open": open,
        "extension": "",
    },
    "gzip": {
        "open": _gzip_open,
        "extension": "gz",
    },
    "bz2": {
        "open": _bz2_open,
        "extension": "bz2",
    },
    "lzma": {
        "open": _lzma_open,
        "extension": "lzma",
    },
}


def get_validated_path(filename, compression_fn=None):
    if compression_fn is not None:
        validated_path = f"{filename}.{compression_fn_map[compression_fn]['extension']}"
    else:
        validated_path = filename
    return validated_path


def get_compression_map():
    return compression_fn_map
