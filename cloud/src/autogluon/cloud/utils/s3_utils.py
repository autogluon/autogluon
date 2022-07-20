import boto3


def download_s3_file(bucket, prefix, path):
    s3 = boto3.client('s3')
    s3.download_file(bucket, prefix, path)
