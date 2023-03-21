import boto3
import logging
import os
import shutil

from typing import List


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
    keep_root_dir: bool = False,
    error_if_exists: bool = True,
    delete_if_exists: bool = False,
    dry_run: bool = False,
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
            └── test2.txt/
                └── temp/
                    └── test3.txt
    If `keep_root_dir` is set to True, then the root directory of the s3 folder will also be downloaded
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
    delete_if_exists: bool
        Whether to delete the root folder and all contents within if the root folder exists already
    dry_run: bool
        Whether to perform the directory creation and file downloading.
        If True, will isntead log every file that will be downloaded and every directory that will be created
    verbose: bool
        Whether to log detailed loggings
    """

    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket)
    objs = list(bucket.objects.filter(Prefix=prefix))
    if not keep_root_dir:
        objs = [obj for obj in objs if obj.key != prefix]
    local_paths = _get_local_paths_to_download_objs_with_common_root(
        s3_objs=[obj.key for obj in objs],
        local_path=local_path,
        keep_root_dir=keep_root_dir
    )
    objs_no_folder = [path for path in local_paths if not path.endswith("/")]
    if verbose:
        logger.log(20, f"Will download {len(objs_no_folder)} objects from s3://{bucket}/{prefix} to {local_path}")
    
    # make directory
    local_rootdir = os.path.commonpath(local_paths)
    if os.path.isdir(local_rootdir) and not dry_run:
        if error_if_exists:
            raise ValueError(f"Directory {local_rootdir} already exsists. Please pass in a different `local_path` or set `error_if_exsits` to `False`")
        if delete_if_exists:
            logger.warning(f"Will delete {local_rootdir} and all its content within because this folder already exsists and `delete_if_exists` = `True`")
            shutil.rmtree(local_rootdir)
    local_obj_dirs = [path for path in local_paths if path.endswith("/")]
    for obj_dir in local_obj_dirs:
        if not dry_run:
            os.makedirs(obj_dir, exist_ok=True)
        else:
            logger.log(20, f"Will create directory {obj_dir}")
    
    for obj, local_path in zip(objs, local_paths):
        if local_path.endswith("/"):
            continue
        if not dry_run:
            bucket.download_file(obj.key, local_path)
        else:
            logger.log(20, f"Will download {obj.key} to {local_path}")
            
            
def _get_local_paths_to_download_objs_with_common_root(s3_objs: List[str], local_path: str, keep_root_dir: bool = False) -> List[str]:
    """
    Generate the local path to download objs to for each object.
    The local path will maintain the folder structure starting the the first sub-layer of common root of the `s3_objs`
    For example, assuming bucket = bar and the bucket structure looks like this
        .
        └── bar/
            ├── test.txt
            └── foo/
                ├── test2.txt
                └── temp/
                    └── test3.txt
    assuming `s3_objs = ["foo/temp/", "foo/temp/test3.txt"]`, `local_path = "."` and `keep_root_dir = False`
    This function will return ["test3.txt"]
    Given the same example, if `keep_root_dir = True`, then the function will return ["./temp/", "./temp/test3.txt"]
    
    Parameters
    ----------
    s3_objs: List[str]
        List of objects to download from s3. This list should have a common prefix (having the same root direcory)
        str ends with "/" are recognized as directory.
    local_path: str
        Local path to download the objects to
    keep_root_dir: bool
        Whether to keep the common root inside the local folder structure
    
    Return
    ------
        A list representing local object paths, including directory, to download to.
    """
    if len(s3_objs) == 0:
        return []
    common_path = os.path.commonpath(s3_objs)
    assert len(common_path) > 0, "Please pass in a list of objects with a common root folder"
    if keep_root_dir:
        irrelevant_common_prefix = os.path.dirname(os.path.normpath(common_path))
    else:
        irrelevant_common_prefix = os.path.normpath(common_path)
    normalized_local_paths = [obj[len(irrelevant_common_prefix)+1:] if len(irrelevant_common_prefix) > 0 else obj for obj in s3_objs]
    normalized_local_paths = [path for path in normalized_local_paths if len(path) > 0]

    return [os.path.join(local_path, path)for path in normalized_local_paths]
