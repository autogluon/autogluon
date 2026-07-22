import os
import stat
import tarfile
import zipfile


def safe_unpack_archive(path: str | os.PathLike, dest_dir: str | os.PathLike) -> None:
    """Safely extract a zip or tar archive, rejecting path traversal and unsafe entries.

    The archive format is auto-detected; supports zip and tar (incl. gz/bz2/xz).
    """
    path = str(path)
    dest_dir = os.path.realpath(str(dest_dir))

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            _validate_zip(zf, dest_dir)
            zf.extractall(dest_dir)  # nosec B202
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            _validate_tar(tf, dest_dir)
            tf.extractall(dest_dir)  # nosec B202
    else:
        raise ValueError(f"Unsupported or unrecognized archive format: {path}")


def _validate_zip(zf: zipfile.ZipFile, dest_dir: str) -> None:
    for member in zf.infolist():
        member_path = os.path.realpath(os.path.join(dest_dir, member.filename))
        if not member_path.startswith(dest_dir + os.sep) and member_path != dest_dir:
            raise ValueError(f"Path traversal detected: {member.filename} would extract outside {dest_dir}")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(f"Archive contains symlink: {member.filename}")


def _validate_tar(tf: tarfile.TarFile, dest_dir: str) -> None:
    for member in tf.getmembers():
        member_path = os.path.realpath(os.path.join(dest_dir, member.name))
        if not member_path.startswith(dest_dir + os.sep) and member_path != dest_dir:
            raise ValueError(f"Path traversal detected: {member.name} would extract outside {dest_dir}")
        if member.issym() or member.islnk():
            raise ValueError(f"Archive contains link: {member.name}")
        if member.isdev() or member.isfifo():
            raise ValueError(f"Archive contains special file: {member.name}")
