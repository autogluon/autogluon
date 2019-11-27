import io
import pickle
import boto3

from . import load_pointer
from .. import s3_utils


def load(path, format=None):
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
        print('Loading', path)
        s3_bucket, s3_prefix = s3_utils.s3_path_to_bucket_prefix(s3_path=path)
        s3 = boto3.resource('s3')
        return pickle.loads(s3.Bucket(s3_bucket).Object(s3_prefix).get()['Body'].read())

    print('Loading', path)
    with open(path, 'rb') as fin:
        object = pickle.load(fin)
    return object


def load_with_fn(path, pickle_fn, format=None):
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
        print('Loading', path)
        s3_bucket, s3_prefix = s3_utils.s3_path_to_bucket_prefix(s3_path=path)
        s3 = boto3.resource('s3')
        # Has to be wrapped in IO buffer since s3 stream does not implement seek()
        buff = io.BytesIO(s3.Bucket(s3_bucket).Object(s3_prefix).get()['Body'].read())
        return pickle_fn(buff)

    print('Loading', path)
    with open(path, 'rb') as fin:
        object = pickle_fn(fin)
    return object
