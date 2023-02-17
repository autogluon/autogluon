import logging
import os
import pathlib

logger = logging.getLogger(__name__)


def list_bucket_s3(bucket):
    import boto3
    logger.log(15, 'Listing s3 bucket: ' + str(bucket))

    s3bucket = boto3.resource('s3')
    my_bucket = s3bucket.Bucket(bucket)
    files = []
    for object in my_bucket.objects.all():
        files.append(object.key)
        logger.log(15, str(object.key))
    return files


def download(input_bucket, input_prefix, local_path):
    import boto3
    directory = os.path.dirname(local_path)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    s3 = boto3.resource('s3')
    s3.Bucket(input_bucket).download_file(input_prefix, local_path)


def list_bucket_prefix_s3(bucket, prefix):
    return list_bucket_prefix_suffix_s3(bucket=bucket, prefix=prefix)


def list_bucket_prefix_suffix_s3(bucket, prefix, suffix=None, banned_suffixes=None):
    import boto3
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
    import boto3
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
