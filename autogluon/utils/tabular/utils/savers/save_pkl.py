# TODO: Standardize / unify this code with ag.save()
import os, pickle, tempfile, logging, boto3, math
import gzip
import bz2
import lzma

from .. import s3_utils

logger = logging.getLogger(__name__)

# TODO: extract to compression helper
compression_fn_map = {
    None: {
        'open': open,
        'extension': 'pkl',
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


def save(path, object, format=None, verbose=True, compression_fn=None, compression_fn_kwargs=None):
    if compression_fn in compression_fn_map:
        validated_path = get_validated_path(path, compression_fn)
    else:
        raise ValueError(f'compression_fn={compression_fn} is not a valid compression_fn. Valid values: {compression_fn_map.keys()}')

    pickle_fn = lambda o, buffer: pickle.dump(o, buffer, protocol=4)
    save_with_fn(validated_path, object, pickle_fn, format=format, verbose=verbose, compression_fn=compression_fn,
                 compression_fn_kwargs=compression_fn_kwargs)


def save_with_fn(path, object, pickle_fn, format=None, verbose=True, compression_fn=None, compression_fn_kwargs=None):
    if verbose:
        logger.log(15, 'Saving '+str(path))
    if s3_utils.is_s3_url(path):
        format = 's3'
    if format == 's3':
        save_s3(path, object, pickle_fn, verbose=verbose)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if compression_fn_kwargs is None:
            compression_fn_kwargs = {}

        with compression_fn_map[compression_fn]['open'](path, 'wb', **compression_fn_kwargs) as fout:
            pickle_fn(object, fout)


def save_s3(path: str, obj, pickle_fn, verbose=True):
    if verbose:
        logger.info(f'save object to {path}')
    with tempfile.TemporaryFile() as f:
        pickle_fn(obj, f)
        f.flush()
        f.seek(0)

        bucket, key = s3_utils.s3_path_to_bucket_prefix(path)
        s3_client = boto3.client('s3')
        try:
            config = boto3.s3.transfer.TransferConfig()   # enable multipart uploading for files larger than 8MB
            response = s3_client.upload_fileobj(f, bucket, key, Config=config)
        except:
            logger.exception('Failed to save object to s3')
            raise


# TODO: extract to compression helper
def get_compression_map():
    return compression_fn_map


# TODO: extract to compression helper
def get_validated_path(filename, compression_fn=None):
    if compression_fn is not None:
        filename_root = os.path.splitext(filename)[0]
        validated_path = f"{filename_root}.{compression_fn_map[compression_fn]['extension']}"
    else:
        validated_path = filename
    return validated_path


# TODO: extract to compression helper OR remove
def _get_hr_filesize(path):
    size_bytes = os.path.getsize(path)
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])