from autogluon.common.utils.s3_utils import _get_local_path_to_download_objs, _get_local_objs_to_upload_and_s3_prefix

import pytest
import tempfile
import os

@pytest.mark.parametrize(
    "s3_objs,prefix,local_path,expected_objs",
    [
        ([], "foo", ".", []),
        (["foo/test.txt"], "foo", ".", ["test.txt"]),
        (["foo/test.txt", "foo/test2.txt"], "foo", ".", ["test.txt", "test2.txt"]),
        (["foo/test.txt"], "foo", ".", ["test.txt"]),
        (["foo/temp/test.txt", "foo/test2.txt"], "foo", ".", ["temp/test.txt", "test2.txt"]),
        (["foo/temp/test.txt"], "foo", ".", ["temp/test.txt"]),
        (["foo/temp/temp2/test.txt"], "foo", ".", ["temp/temp2/test.txt"]),
        (["foo/temp/temp2/test.txt", "foo/temp/temp3/test.txt"], "foo", ".", ["temp/temp2/test.txt", "temp/temp3/test.txt"]),
    ]
)
def test_get_local_path_to_download_objs(s3_objs, prefix, local_path, expected_objs):
    objs = _get_local_path_to_download_objs(s3_objs=s3_objs, prefix=prefix, local_path=local_path)
    assert sorted(objs) == sorted(expected_objs)

    
def test_get_local_objs_to_upload_and_s3_prefix():
    """
    The following code tests such a folder structure:
    .
    └── temp_dir/
        ├── test.txt
        └── dir1/
            ├── dir2
            └── test2.txt
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        test1 = "test.txt"
        open(test1, "a").close()
        dir1 = "dir1"
        dir2 = "dir2"
        os.makedirs(os.path.join(dir1, dir2))
        test2 = os.path.join(dir1, "test2.txt")
        open(test2, "a").close()
        result = _get_local_objs_to_upload_and_s3_prefix(folder_to_upload=temp_dir)
        print(result)
        assert (os.path.join(temp_dir, test1), test1) in result
        assert (os.path.join(temp_dir, test2), test2) in result
        assert len(result) == 2
        

def test_get_local_objs_to_upload_and_s3_prefix_empty():
    with tempfile.TemporaryDirectory() as temp_dir:
        result = _get_local_objs_to_upload_and_s3_prefix(folder_to_upload=temp_dir)
        assert len(result) == 0
