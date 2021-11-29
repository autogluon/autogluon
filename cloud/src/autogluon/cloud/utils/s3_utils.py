import boto3


def is_s3_url(path):
    if type(path) != str:
        return False
    if (path[:2] == 's3') and ('://' in path[:6]):
        return True
    return False


def s3_path_to_bucket_prefix(s3_path):
    s3_path_cleaned = s3_path.split('://', 1)[1]
    bucket, prefix = s3_path_cleaned.split('/', 1)

    # print('extracted bucket:', bucket, 'and prefix:', prefix, 'from s3_path:', s3_path)
    return bucket, prefix


def s3_bucket_prefix_to_path(bucket, prefix, version='s3'):
    return version + '://' + bucket + '/' + prefix


def download_s3_file(bucket, prefix, path):
    s3 = boto3.client('s3')
    s3.download_file(bucket, prefix, path)
