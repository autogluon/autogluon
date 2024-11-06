import os
import tempfile

import pytest

from autogluon.common.utils.s3_utils import (
    _get_local_objs_to_upload_and_s3_prefix,
    _get_local_path_to_download_objs,
    get_s3_to_local_tuple_list,
)


@pytest.mark.parametrize(
    "s3_objs,prefix,local_path,expected_objs",
    [
        ([], "foo", ".", []),
        (["foo/test.txt"], "foo", ".", ["test.txt"]),
        (["foo/test.txt", "foo/test2.txt"], "foo", ".", ["test.txt", "test2.txt"]),
        (["foo/test.txt"], "foo", ".", ["test.txt"]),
        (["foo/temp/test.txt", "foo/test2.txt"], "foo", ".", [os.path.join("temp", "test.txt"), "test2.txt"]),
        (["foo/temp/test.txt"], "foo", ".", [os.path.join("temp", "test.txt")]),
        (["foo/temp/temp2/test.txt"], "foo", ".", [os.path.join("temp", "temp2", "test.txt")]),
        (
            ["foo/temp/temp2/test.txt", "foo/temp/temp3/test.txt"],
            "foo",
            ".",
            [os.path.join("temp", "temp2", "test.txt"), os.path.join("temp", "temp3", "test.txt")],
        ),
    ],
)
def test_get_local_path_to_download_objs(s3_objs, prefix, local_path, expected_objs):
    objs = _get_local_path_to_download_objs(s3_objs=s3_objs, prefix=prefix, local_path=local_path)
    assert objs == expected_objs


@pytest.mark.parametrize(
    "s3_bucket,s3_prefix,local_path,s3_prefixes,expected_output",
    [
        ("foo", "bar", "local/path", [], []),
        (
            "foo",
            "bar",
            "local/path",
            ["bar/test.txt"],
            [("s3://foo/bar/test.txt", os.path.join("local", "path", "test.txt"))],
        ),
        ("foo", "", "", ["test.txt"], [("s3://foo/test.txt", "test.txt")]),
        (
            "foo",
            "bar/bar",
            "local/path",
            ["bar/bar/test.txt"],
            [("s3://foo/bar/bar/test.txt", os.path.join("local", "path", "test.txt"))],
        ),
        (
            "foo",
            "bar",
            "local/path",
            ["bar/temp/test.txt", "bar/test2.txt"],
            [
                ("s3://foo/bar/temp/test.txt", os.path.join("local", "path", "temp", "test.txt")),
                ("s3://foo/bar/test2.txt", os.path.join("local", "path", "test2.txt")),
            ],
        ),
        (
            "foo",
            "bar",
            "local/path",
            ["bar/temp/test.txt"],
            [("s3://foo/bar/temp/test.txt", os.path.join("local", "path", "temp", "test.txt"))],
        ),
        ("foo", "", "", ["a"], [("s3://foo/a", "a")]),
    ],
)
def test_get_s3_to_local_tuple_list(s3_bucket, s3_prefix, local_path, s3_prefixes, expected_output):
    actual_output = get_s3_to_local_tuple_list(
        s3_bucket=s3_bucket, s3_prefix=s3_prefix, local_path=local_path, s3_prefixes=s3_prefixes
    )
    assert actual_output == expected_output


@pytest.mark.parametrize(
    "s3_bucket,s3_prefix,local_path,s3_prefixes,expected_error",
    [
        ("foo", "bar", "local/path", ["foo/test.txt"], ValueError),
        ("", "bar", "local/path", ["bar/test.txt"], ValueError),
        ("foo", "bar", "local/path", ["foo/test.txt", "bar/test.txt"], ValueError),
    ],
)
def test_get_s3_to_local_tuple_list_raises(s3_bucket, s3_prefix, local_path, s3_prefixes, expected_error):
    with pytest.raises(expected_error):
        get_s3_to_local_tuple_list(
            s3_bucket=s3_bucket, s3_prefix=s3_prefix, local_path=local_path, s3_prefixes=s3_prefixes
        )


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
        old_location = os.getcwd()
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
        os.chdir(old_location)  # Windows will fail if chdir to tempdir https://bugs.python.org/issue42796


def test_get_local_objs_to_upload_and_s3_prefix_empty():
    with tempfile.TemporaryDirectory() as temp_dir:
        result = _get_local_objs_to_upload_and_s3_prefix(folder_to_upload=temp_dir)
        assert len(result) == 0
