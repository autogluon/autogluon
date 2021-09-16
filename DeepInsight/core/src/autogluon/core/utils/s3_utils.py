import boto3


def is_s3_url(path):
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


def delete_s3_prefix(bucket, prefix):
    s3 = boto3.resource('s3')
    objects_to_delete = s3.meta.client.list_objects(Bucket=bucket, Prefix=prefix)

    delete_keys = {'Objects': []}
    delete_keys['Objects'] = [{'Key': k} for k in [obj['Key'] for obj in objects_to_delete.get('Contents', [])]]

    # print(delete_keys)

    if len(delete_keys['Objects']) != 0:
      s3.meta.client.delete_objects(Bucket=bucket, Delete=delete_keys)
