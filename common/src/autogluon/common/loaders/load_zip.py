import logging
import os
import stat

logger = logging.getLogger(__name__)


def safe_extractall(zf, dest_dir):
    """Extract zip file with path traversal and symlink validation."""
    dest_dir = os.path.realpath(dest_dir)
    for member in zf.infolist():
        member_path = os.path.realpath(os.path.join(dest_dir, member.filename))
        if not member_path.startswith(dest_dir + os.sep) and member_path != dest_dir:
            raise ValueError(f"Zip Slip detected: {member.filename} would extract outside {dest_dir}")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(f"Zip contains symlink: {member.filename}")
    zf.extractall(dest_dir)


def unzip(path, sha1sum=None, unzip_dir=None):
    """Unzip a .zip file from path to unzip_dir."""
    from ._utils import download, protected_zip_extraction

    local_file = unzip_dir + os.sep + "file.zip"
    download(path, path=local_file, sha1_hash=sha1sum)

    logger.log(20, f"Unzipping {local_file} to {unzip_dir}")

    protected_zip_extraction(local_file, sha1_hash=sha1sum, folder=unzip_dir)

    return
