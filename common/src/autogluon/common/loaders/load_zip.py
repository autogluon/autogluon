import logging
import os

logger = logging.getLogger(__name__)


def unzip(path, sha1sum=None, unzip_dir=None):
    """Unzip a .zip file from path to unzip_dir."""
    from ._utils import download, protected_zip_extraction

    local_file = unzip_dir + os.sep + "file.zip"
    download(path, path=local_file, sha1_hash=sha1sum)

    logger.log(20, f"Unzipping {local_file} to {unzip_dir}")

    protected_zip_extraction(local_file, sha1_hash=sha1sum, folder=unzip_dir)

    return
