from autogluon.common.utils.s3_utils import _get_local_path_to_download_objs_and_local_dir_to_create

import pytest

@pytest.mark.parametrize(
    "s3_objs,prefix,local_path,expected_objs,expected_dirs",
    [
        (["foo/"], "foo", ".", [], []),
        (["foo/", "foo/test.txt"], "foo", ".", ["test.txt"], []),
        (["foo/test.txt", "foo/test2.txt"], "foo", ".", ["test.txt", "test2.txt"], []),
        (["foo/temp/", "foo/test.txt"], "foo", ".", ["test.txt"], ["temp"]),
        (["foo/temp/test.txt", "foo/test2.txt"], "foo", ".", ["temp/test.txt", "test2.txt"], ["temp"]),
        (["foo/temp/test.txt"], "foo", ".", ["temp/test.txt"], ["temp"]),
        (["foo/temp/temp2/test.txt"], "foo", ".", ["temp/temp2/test.txt"], ["temp/temp2"]),
        (["foo/temp/temp2/test.txt", "foo/temp/temp3/test.txt"], "foo", ".", ["temp/temp2/test.txt", "temp/temp3/test.txt"], ["temp/temp2", "temp/temp3"]),
    ]
)
def test_get_local_path_to_download_objs_and_local_dir_to_create(s3_objs, prefix, local_path, expected_objs, expected_dirs):
    objs, dirs = _get_local_path_to_download_objs_and_local_dir_to_create(s3_objs=s3_objs, prefix=prefix, local_path=local_path)
    assert sorted(objs) == sorted(expected_objs)
    assert sorted(dirs) == sorted(expected_dirs)
