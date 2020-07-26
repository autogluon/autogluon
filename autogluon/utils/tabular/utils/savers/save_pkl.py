# TODO: Standardize / unify this code with ag.save()
import os, pickle, tempfile, logging, boto3, math, gzip

from .. import s3_utils

logger = logging.getLogger(__name__)


def save(path, object, format=None, verbose=True, compression_level=0):
    pickle_fn = lambda o, buffer: pickle.dump(o, buffer, protocol=4)
    save_with_fn(path, object, pickle_fn, format=format, verbose=verbose, compression_level=compression_level)


def save_with_fn(path, object, pickle_fn, format=None, verbose=True, compression_level=0):
    if verbose:
        logger.log(15, 'Saving '+str(path))
    if s3_utils.is_s3_url(path):
        format = 's3'
    if format == 's3':
        save_s3(path, object, pickle_fn, verbose=verbose)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # TODO: refactor/remove - temp testing
        if compression_level == 0:
            with open(path, 'wb') as fout:
                pickle_fn(object, fout)
            print("-------------------------")
            print(f"using compression_level {compression_level}..")
            print(f"saved {path}..")
            print(f"file size without compression: {_get_hr_filesize(path)}")
            print("-------------------------")
        else:
            # compressed_path = f"{path}.gz"
            compressed_path = path
            with gzip.open(compressed_path, 'wb', compresslevel=compression_level) as compressed_fout:
                pickle_fn(object, compressed_fout)
            print("-------------------------")
            print(f"using compression_level {compression_level}..")
            print(f"saved compressed {compressed_path}..")
            print(f"file size with compression: {_get_hr_filesize(compressed_path)}")
            print("-------------------------")


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

# TODO: refactor/remove - temp testing
def _get_hr_filesize(path):
    size_bytes = os.path.getsize(path)
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])