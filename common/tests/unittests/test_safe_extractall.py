import io
import os
import tarfile
import tempfile
import zipfile

import pytest

from autogluon.common.loaders.load_archive import safe_unpack_archive


def test_normal_zip_extracts_successfully():
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data/file.txt", "hello")
            zf.writestr("data/nested/file2.txt", "world")

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        safe_unpack_archive(zip_path, extract_dir)

        assert os.path.exists(os.path.join(extract_dir, "data", "file.txt"))
        assert os.path.exists(os.path.join(extract_dir, "data", "nested", "file2.txt"))


def test_zip_path_traversal_blocked():
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../../etc/passwd", "pwned")

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_unpack_archive(zip_path, extract_dir)


def test_zip_symlink_blocked():
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            info = zipfile.ZipInfo("link")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "/etc")

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        with pytest.raises(ValueError, match="Archive contains symlink"):
            safe_unpack_archive(zip_path, extract_dir)


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
        safe_unpack_archive(tar_path, extract_dir)

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
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_unpack_archive(tar_path, extract_dir)


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
        with pytest.raises(ValueError, match="Archive contains link"):
            safe_unpack_archive(tar_path, extract_dir)


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
        with pytest.raises(ValueError, match="Archive contains link"):
            safe_unpack_archive(tar_path, extract_dir)


def test_tar_fifo_blocked():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, "test.tar")
        with tarfile.open(tar_path, "w") as tf:
            info = tarfile.TarInfo("fifo")
            info.type = tarfile.FIFOTYPE
            tf.addfile(info)

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        with pytest.raises(ValueError, match="Archive contains special file"):
            safe_unpack_archive(tar_path, extract_dir)


def test_tar_device_blocked():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, "test.tar")
        with tarfile.open(tar_path, "w") as tf:
            info = tarfile.TarInfo("device")
            info.type = tarfile.CHRTYPE
            info.devmajor = 1
            info.devminor = 3
            tf.addfile(info)

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        with pytest.raises(ValueError, match="Archive contains special file"):
            safe_unpack_archive(tar_path, extract_dir)


def test_unsupported_format_rejected():
    with tempfile.TemporaryDirectory() as tmp_dir:
        bad_path = os.path.join(tmp_dir, "file.txt")
        with open(bad_path, "w") as f:
            f.write("not an archive")

        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir)
        with pytest.raises(ValueError, match="Unsupported or unrecognized archive format"):
            safe_unpack_archive(bad_path, extract_dir)
