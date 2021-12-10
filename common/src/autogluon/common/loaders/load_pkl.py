import io, logging, pickle, boto3

from ..loaders import load_pointer
from ..utils import compression_utils, s3_utils

logger = logging.getLogger(__name__)


def load(path, format=None, verbose=True, **kwargs):
    compression_fn = kwargs.get('compression_fn', None)
    compression_fn_kwargs = kwargs.get('compression_fn_kwargs', None)

    if path.endswith('.pointer'):
        format = 'pointer'
    elif s3_utils.is_s3_url(path):
        format = 's3'
    if format == 'pointer':
        content_path = load_pointer.get_pointer_content(path)
        if content_path == path:
            raise RecursionError('content_path == path! : ' + str(path))
        return load(path=content_path)
    elif format == 's3':
        if verbose: logger.log(15, 'Loading: %s' % path)
        s3_bucket, s3_prefix = s3_utils.s3_path_to_bucket_prefix(s3_path=path)
        s3 = boto3.resource('s3')
        return pickle.loads(s3.Bucket(s3_bucket).Object(s3_prefix).get()['Body'].read())

    if verbose: logger.log(15, 'Loading: %s' % path)

    compression_fn_map = compression_utils.get_compression_map()
    validated_path = compression_utils.get_validated_path(path, compression_fn=compression_fn)

    if compression_fn_kwargs is None:
        compression_fn_kwargs = {}

    if compression_fn in compression_fn_map:
        with compression_fn_map[compression_fn]['open'](validated_path, 'rb', **compression_fn_kwargs) as fin:
            object = pickle.load(fin)
    else:
        raise ValueError(f'compression_fn={compression_fn} or compression_fn_kwargs={compression_fn_kwargs} are not valid. Valid function values: {compression_fn_map.keys()}')

    return object


def load_with_fn(path, pickle_fn, format=None, verbose=True):
    if path.endswith('.pointer'):
        format = 'pointer'
    elif s3_utils.is_s3_url(path):
        format = 's3'
    if format == 'pointer':
        content_path = load_pointer.get_pointer_content(path)
        if content_path == path:
            raise RecursionError('content_path == path! : ' + str(path))
        return load_with_fn(content_path, pickle_fn)
    elif format == 's3':
        if verbose: logger.log(15, 'Loading: %s' % path)
        s3_bucket, s3_prefix = s3_utils.s3_path_to_bucket_prefix(s3_path=path)
        s3 = boto3.resource('s3')
        # Has to be wrapped in IO buffer since s3 stream does not implement seek()
        buff = io.BytesIO(s3.Bucket(s3_bucket).Object(s3_prefix).get()['Body'].read())
        return pickle_fn(buff)

    if verbose: logger.log(15, 'Loading: %s' % path)
    with open(path, 'rb') as fin:
        object = pickle_fn(fin)
    return object
