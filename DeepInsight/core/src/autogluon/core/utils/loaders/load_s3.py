import boto3, os, pathlib, logging

from . import load_pd
from .. import s3_utils

logger = logging.getLogger(__name__)

def list_bucket_s3(bucket):
    logger.log(15, 'Listing s3 bucket: '+str(bucket))

    s3bucket = boto3.resource('s3')
    my_bucket = s3bucket.Bucket(bucket)
    files = []
    for object in my_bucket.objects.all():
        files.append(object.key)
        logger.log(15, str(object.key))
    return files


def download(input_bucket, input_prefix, local_path):
    directory = os.path.dirname(local_path)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    s3 = boto3.resource('s3')
    s3.Bucket(input_bucket).download_file(input_prefix, local_path)


def list_bucket_prefix_s3(bucket, prefix):
    return list_bucket_prefix_suffix_s3(bucket=bucket, prefix=prefix)


def list_bucket_prefix_suffix_s3(bucket, prefix, suffix=None, banned_suffixes=None):
    if banned_suffixes is None:
        banned_suffixes = []
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucket)
    prefix = prefix

    files = []
    for object_summary in my_bucket.objects.filter(Prefix=prefix):
        suffix_full = object_summary.key.split(prefix, 1)[1]
        is_banned = False
        for banned_suffix in banned_suffixes:
            if banned_suffix in suffix_full:
                is_banned = True
        if (not is_banned) and ((suffix is None) or (suffix in suffix_full)):
            files.append(object_summary.key)

    return files


def list_bucket_prefix_suffix_contains_s3(bucket, prefix, suffix=None, banned_suffixes=None, contains=None):
    if banned_suffixes is None:
        banned_suffixes = []
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucket)
    prefix = prefix

    files = []
    for object_summary in my_bucket.objects.filter(Prefix=prefix):
        suffix_full = object_summary.key.split(prefix, 1)[1]
        is_banned = False
        for banned_suffix in banned_suffixes:
            if banned_suffix in suffix_full:
                is_banned = True
        if (not is_banned) and ((suffix is None) or (suffix in suffix_full)) and (contains is None or contains in suffix_full):
            files.append(object_summary.key)

    return files


def load_multipart_s3(bucket, prefix, columns_to_keep=None, dtype=None, sample_count=None):
    files = list_bucket_prefix_s3(bucket, prefix)
    files_cleaned = [file for file in files if prefix + '/part-' in file]
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version='s3') for file in files_cleaned]
    if sample_count is not None:
        logger.log(15, 'Taking sample of '+str(sample_count)+' of '+str(len(paths_full))+' s3 files to load')
        paths_full = paths_full[:sample_count]

    df = load_pd.load(path=paths_full, columns_to_keep=columns_to_keep, dtype=dtype)
    return df
