import io
import os
import tarfile
import tempfile
import zipfile

import pytest

from autogluon.multimodal.cli.prepare_detection_dataset import _safe_tar_extractall, unpack


def test_normal_tar_extracts_successfully():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, "test.tar")
        with tarfile.open(tar_path, "w") as tf:
            data = b"hello"
            info = tarfile.TarInfo("data/file.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        with tarfile.open(tar_path) as tf:
            _safe_tar_extractall(tf, extract_dir)

        assert os.path.exists(os.path.join(extract_dir, "data", "file.txt"))


def test_tar_path_traversal_blocked():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, "test.tar")
        with tarfile.open(tar_path, "w") as tf:
            data = b"pwned"
            info = tarfile.TarInfo("../../etc/passwd")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        with tarfile.open(tar_path) as tf:
            with pytest.raises(ValueError, match="Tar Slip detected"):
                _safe_tar_extractall(tf, extract_dir)


def test_tar_symlink_blocked():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, "test.tar")
        with tarfile.open(tar_path, "w") as tf:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc"
            tf.addfile(info)

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        with tarfile.open(tar_path) as tf:
            with pytest.raises(ValueError, match="Tar contains link"):
                _safe_tar_extractall(tf, extract_dir)


def test_tar_hardlink_blocked():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, "test.tar")
        with tarfile.open(tar_path, "w") as tf:
            info = tarfile.TarInfo("link")
            info.type = tarfile.LNKTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        with tarfile.open(tar_path) as tf:
            with pytest.raises(ValueError, match="Tar contains link"):
                _safe_tar_extractall(tf, extract_dir)


def test_unpack_zip_uses_safe_extraction():
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data/file.txt", "hello")

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        unpack([zip_path], extract_dir)

        assert os.path.exists(os.path.join(extract_dir, "data", "file.txt"))
