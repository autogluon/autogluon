from autogluon.common.utils.s3_utils import _get_local_paths_to_download_objs_with_common_root

import pytest

@pytest.mark.parametrize(
    "s3_objs,keep_root_dir,expected_value",
    [
        ([], True, []),
        ([], False, []),
        (["foo/"], True, ["./foo/"]),
        (["foo/"], False, []),
        (["foo/temp/"], True, ["./temp/"]),
        (["foo/temp/"], False, []),
        (["foo/temp/", "foo/temp/test.txt"], True, ["./temp/", "./temp/test.txt"]),
        (["foo/temp/", "foo/temp/test.txt"], False, ["./test.txt"]),
        (["foo/temp/", "foo/test/"], True, ["./foo/temp/", "./foo/test/"]),
        (["foo/temp/", "foo/test/"], False, ["./temp/", "./test/"]),
        (["bar/foo/", "bar/foo/temp/test.txt", "bar/foo/test2.txt"], True, ["./foo/", "./foo/temp/test.txt", "./foo/test2.txt"]),
        (["bar/foo/", "bar/foo/temp/test.txt", "bar/foo/test2.txt"], False, ["./temp/test.txt", "./test2.txt"]),
        (["foo/", "bar/"], True, "raise"),
        (["foo/", "bar/"], False, "raise")
    ]
)
def test_get_local_paths_to_download_objs_with_common_root(s3_objs, keep_root_dir, expected_value):
    if not expected_value == "raise":
        paths = _get_local_paths_to_download_objs_with_common_root(s3_objs=s3_objs, local_path=".", keep_root_dir=keep_root_dir)
        assert paths == expected_value
    else:
        with pytest.raises(Exception):
            _get_local_paths_to_download_objs_with_common_root(s3_objs=s3_objs, local_path=".", keep_root_dir=keep_root_dir)
