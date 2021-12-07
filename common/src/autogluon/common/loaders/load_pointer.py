import os, boto3, logging

from ..utils import s3_utils

logger = logging.getLogger(__name__)

def get_pointer_content(path, verbose=True):
    if s3_utils.is_s3_url(path):
        bucket, key = s3_utils.s3_path_to_bucket_prefix(path)
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket, key)
        content_path = obj.get()['Body'].read().decode('utf-8')
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f = open(path, "r")
        content_path = f.read()
        f.close()
    if verbose:
        logger.log(15, 'Loaded pointer file '+str(path)+' pointing to '+str(content_path))

    return content_path
