import os
import tempfile
import zipfile

import pytest

from autogluon.common.loaders.load_zip import safe_extractall


def test_normal_zip_extracts_successfully():
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data/file.txt", "hello")
            zf.writestr("data/nested/file2.txt", "world")

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            safe_extractall(zf, extract_dir)

        assert os.path.exists(os.path.join(extract_dir, "data", "file.txt"))
        assert os.path.exists(os.path.join(extract_dir, "data", "nested", "file2.txt"))


def test_path_traversal_blocked():
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../../etc/passwd", "pwned")

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            with pytest.raises(ValueError, match="Zip Slip detected"):
                safe_extractall(zf, extract_dir)


def test_absolute_path_blocked():
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("/tmp/file.txt", "pwned")

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            with pytest.raises(ValueError, match="Zip Slip detected"):
                safe_extractall(zf, extract_dir)


def test_symlink_blocked():
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            info = zipfile.ZipInfo("link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "/etc")

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            with pytest.raises(ValueError, match="Zip contains symlink"):
                safe_extractall(zf, extract_dir)
