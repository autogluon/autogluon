import gzip
import bz2
import lzma


compression_fn_map = {
    None: {
        'open': open,
        'extension': '',
    },
    'gzip': {
        'open': gzip.open,
        'extension': 'gz',
    },
    'bz2': {
        'open': bz2.open,
        'extension': 'bz2',
    },
    'lzma': {
        'open': lzma.open,
        'extension': 'lzma',
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
