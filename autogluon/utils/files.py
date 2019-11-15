import os
import requests
import errno
import shutil
import hashlib
import zipfile
import logging
from .tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ['unzip', 'download', 'mkdir', 'check_sha1', 'unzip', 'raise_num_file']

def unzip(zip_file_path, root=os.path.expanduser('./')):
    """Unzip the files.
    """
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)

def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL

    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        logger.info('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.

    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def mkdir(path):
    """make dir exists okay"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def raise_num_file(nofile_atleast=4096):
    try:
        import resource as res
    except ImportError: #Windows
        res = None
    if res is None:
        return (None,)*2
    # what is current ulimit -n setting?
    soft,ohard = res.getrlimit(res.RLIMIT_NOFILE)
    hard = ohard
    # increase limit (soft and even hard) if needed
    if soft < nofile_atleast:
        soft = nofile_atleast

        if hard<soft:
            hard = soft

        #logger.warning('setting soft & hard ulimit -n {} {}'.format(soft,hard))
        try:
            res.setrlimit(res.RLIMIT_NOFILE,(soft,hard))
        except (ValueError,res.error):
            try:
               hard = soft
               logger.warning('trouble with max limit, retrying with soft,hard {},{}'.format(soft,hard))
               res.setrlimit(res.RLIMIT_NOFILE,(soft,hard))
            except Exception:
               logger.warning('failed to set ulimit')
               soft,hard = res.getrlimit(res.RLIMIT_NOFILE)

    return soft,hard

raise_num_file()
