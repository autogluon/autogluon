import logging


logger = logging.getLogger(__name__)


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
    import boto3
    s3 = boto3.resource('s3')
    objects_to_delete = s3.meta.client.list_objects(Bucket=bucket, Prefix=prefix)

    delete_keys = {'Objects': []}
    delete_keys['Objects'] = [{'Key': k} for k in [obj['Key'] for obj in objects_to_delete.get('Contents', [])]]

    # print(delete_keys)

    if len(delete_keys['Objects']) != 0:
        s3.meta.client.delete_objects(Bucket=bucket, Delete=delete_keys)
        

def download_s3_folder(
    bucket: str,
    prefix: str,
    local_path: str,
    error_if_exists: bool = True,
    verbose: bool = True
):
    """
    This util function downloads a s3 folder and maintain its structure.
    For example, assuming bucket = bar and prefix = foo, and the bucket structure looks like this
        .
        └── bar/
            ├── test.txt
            └── foo/
                ├── test2.txt
                └── temp/
                    └── test3.txt
    This util will download foo to `local_path` and maintain the structure:
        .
        └── local_path/
            └── foo/
                ├── test2.txt
                └── temp/
                    └── test3.txt
                    
    Parameters
    ----------
    bucket: str
        The name of the bucket
    prefix: str
        The prefix of the object/folder to be downloaded
    local_path: str
        The local path to download the object/folder into
    error_if_exists: bool
        Whether to raise an error if the root folder exists already
    verbose: bool
        Whether to log detailed loggings
    """
    import boto3
    import os

    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket)
    objs = list(bucket.objects.filter(Prefix=prefix))
    irrelevent_dirname = os.path.dirname(os.path.normpath(prefix))
    objs_no_folder = [obj for obj in objs if not obj.key.endswith("/")]
    logger.log(20, f"Will download {len(objs_no_folder)} objects from s3://{bucket}/{prefix} to {local_path}")

    for obj in objs:
        # remove the file name from the object key and remove irrelevent parent folder along the path
        obj_dir = os.path.relpath(os.path.dirname(obj.key), irrelevent_dirname)
        obj_dir = os.path.join(local_path, obj_dir)
        # get the full object path
        obj_path = os.path.join(obj_dir, os.path.basename(obj.key))

        # create nested directory structure
        os.makedirs(obj_dir, exist_ok=True)

        # save file with full path locally
        if not obj.key.endswith("/"):
            bucket.download_file(obj.key, obj_path)
