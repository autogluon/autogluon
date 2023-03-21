from autogluon.common.utils.s3_utils import _get_local_paths_to_download_objs_with_common_root

import pytest

@pytest.mark.parametrize(
    "s3_objs,expected_value",
    [
        ([], []),
        (["foo/temp/"], ["./temp/"]),
        (["foo/temp/", "foo/temp/test.txt"], ["./temp/", "./temp/test.txt"]),
        (["foo/temp/", "foo/test/"], ["./foo/temp/", "./foo/test/"]),
        (["bar/foo/", "bar/foo/temp/test.txt", "bar/foo/test2.txt"], ["./foo/", "./foo/temp/test.txt", "./foo/test2.txt"]),
        (["foo/", "bar/"], "raise")
    ]
)
def test_get_local_paths_to_download_objs_with_common_root(s3_objs, expected_value):
    if not expected_value == "raise":
        paths = _get_local_paths_to_download_objs_with_common_root(s3_objs=s3_objs, local_path=".")
        assert paths == expected_value
    else:
        with pytest.raises(Exception):
            _get_local_paths_to_download_objs_with_common_root(s3_objs=s3_objs, local_path=".")
