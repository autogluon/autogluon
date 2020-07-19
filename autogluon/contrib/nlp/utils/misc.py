import os
import sys
import inspect
import logging
import warnings
import functools
import uuid
from types import ModuleType
from typing import Optional, Tuple, Dict
import numpy as np
import hashlib
import requests
import itertools
import random
try:
    import tqdm
except ImportError:
    tqdm = None
from .lazy_imports import try_import_boto3
from collections import OrderedDict
import glob as _glob


S3_PREFIX = 's3://'


def glob(url, separator=','):
    """Return a list of paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards.
    Input may also include multiple patterns, separated by separator.

    Parameters
    ----------
    url : str
        The name of the files
    separator : str, default is ','
        The separator in url to allow multiple patterns in the input
    """
    patterns = [url] if separator is None else url.split(separator)
    result = []
    for pattern in patterns:
        result.extend(_glob.glob(os.path.expanduser(pattern.strip())))
    return result


class AverageSGDTracker(object):
    def __init__(self, params=None):
        """Maintain a set of shadow variables "v" that is calculated by

            v[:] = (1 - 1/t) v + 1/t \theta

        The t is the number of training steps.

        It is also known as "Polyak-Rupert averaging" applied to SGD and was rediscovered in
        "Towards Optimal One Pass Large Scale Learning withAveraged Stochastic Gradient Descent"
         Wei Xu (2011).

        The idea is to average the parameters obtained by stochastic gradient descent.


        Parameters
        ----------
        params : ParameterDict
            The parameters that we are going to track.
        """
        self._track_params = None
        self._average_params = None
        self._initialized = False
        self._n_steps = 0
        if params is not None:
            self.apply(params)

    @property
    def n_steps(self):
        return self._n_steps

    @property
    def average_params(self):
        return self._average_params

    @property
    def initialized(self):
        return self._initialized

    def apply(self, params):
        """ Tell the moving average tracker which parameters we are going to track.

        Parameters
        ----------
        params : ParameterDict
            The parameters that we are going to track and calculate the moving average.
        """
        assert self._track_params is None, 'The MovingAverageTracker is already initialized and'\
                                           ' is not allowed to be initialized again. '
        self._track_params = params
        self._n_steps = 0

    def step(self):
        from mxnet.gluon.utils import shape_is_known
        assert self._track_params is not None, 'You will need to use `.apply(params)`' \
                                               ' to initialize the MovingAverageTracker.'
        for k, v in self._track_params.items():
            assert shape_is_known(v.shape),\
                'All shapes of the tracked parameters must be given.' \
                ' The shape of {} is {}, and it has not been fully initialized.' \
                ' You should call step after the first forward of the model.'.format(k, v.shape)
        ctx = self._track_params.list_ctx()[0]
        if self._average_params is None:
            self._average_params = OrderedDict([(k, v.data(ctx).copy()) for k, v in self._track_params.items()])
        self._n_steps += 1
        decay = 1.0 / self._n_steps
        for name, average_param in self._average_params.items():
            average_param += decay * (self._track_params[name].data(ctx) - average_param)

    def copy_back(self, params=None):
        """ Copy the average parameters back to the given parameters

        Parameters
        ----------
        params : ParameterDict
            The parameters that we will copy tha average params to.
            If it is not given, the tracked parameters will be updated

        """
        if params is None:
            params = self._track_params
        for k, v in params.items():
            v.set_data(self._average_params[k])


def file_line_number(path: str) -> int:
    """

    Parameters
    ----------
    path

    Returns
    -------
    ret
    """
    ret = 0
    with open(path, 'rb') as f:
        for _ in f:
            ret += 1
        return ret


def md5sum(filename):
    with open(filename, mode='rb') as f:
        d = hashlib.md5()
        for buf in iter(functools.partial(f.read, 1024*100), b''):
            d.update(buf)
    return d.hexdigest()


def sha1sum(filename):
    with open(filename, mode='rb') as f:
        d = hashlib.sha1()
        for buf in iter(functools.partial(f.read, 1024*100), b''):
            d.update(buf)
    return d.hexdigest()

def naming_convention(file_dir, file_name):
    """Rename files with 8-character hash"""
    long_hash = sha1sum(os.path.join(file_dir, file_name))
    file_prefix, file_sufix = file_name.split('.')
    new_name = '{file_prefix}-{short_hash}.{file_sufix}'.format(
        file_prefix=file_prefix,
        short_hash=long_hash[:8],
        file_sufix=file_sufix)
    return new_name, long_hash

def logging_config(folder: Optional[str] = None,
                   name: Optional[str] = None,
                   level: int = logging.INFO,
                   console_level: int = logging.INFO,
                   console: bool = True) -> str:
    """Config the logging module"""
    if name is None:
        name = inspect.stack()[1][1].split('.')[0]
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    # Remove all the current handlers
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".log")
    print("All Logs will be saved to {}".format(logpath))
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    if console:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


# TODO(sxjscience) Consider to move it into the official MXNet gluon package
#  Also currently we have not printed the grad_req flag in Parameters, i.e.,
#  print(net.collect_params()) will not print the grad_req flag.
def count_parameters(params) -> Tuple[int, int]:
    """

    Parameters
    ----------
    params


    Returns
    -------
    num_params
        The number of parameters that requires gradient
    num_fixed_params
        The number of parameters that does not require gradient
    """
    # TODO(sxjscience), raise warning if there are -1/0s in the parameters
    num_params = 0
    num_fixed_params = 0
    for k, v in params.items():
        if v.grad_req != 'null':
            if v._data is None:
                warnings.warn('"{}" is not initialized! The total parameter count '
                              'will not be correct.'.format(k))
            else:
                num_params += np.prod(v.shape)
        else:
            if v._data is None:
                warnings.warn('"{}" is not initialized! The total fixed parameter count '
                              'will not be correct.'.format(k))
            else:
                num_fixed_params += np.prod(v.shape)
    return num_params, num_fixed_params


def set_seed(seed):
    import mxnet as mx
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return '{:.1f} {}{}'.format(num, unit, suffix)
        num /= 1024.0
    return '{:.1f} {}{}'.format(num, 'Yi', suffix)


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def repeat(iterable, count=None):
    if count is None:
        while True:
            for sample in iterable:
                yield sample
    else:
        for i in range(count):
            for sample in iterable:
                yield sample


def parse_ctx(data_str):
    import mxnet as mx
    if data_str == '-1' or data_str == '':
        ctx_l = [mx.cpu()]
    elif data_str.startswith('cpu'):
        ctx_l = [mx.cpu()]
    elif data_str.startswith('gpu'):
        ctx_l = [mx.gpu(int(x[len('gpu'):])) for x in data_str.split(',')]
    else:
        ctx_l = [mx.gpu(int(x)) for x in data_str.split(',')]
    return ctx_l


def load_checksum_stats(path: str) -> Dict[str, str]:
    """

    Parameters
    ----------
    path
        Path to the stored checksum

    Returns
    -------
    file_stats
        The file statistics
        url --> sha1_sum
    """
    file_stats = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            name, hex_hash, file_size = line.strip().split()
            file_stats[name] = hex_hash
    return file_stats


class GoogleDriveDownloader:
    """
    Minimal class to download shared files from Google Drive.

    Based on: https://github.com/ndrplz/google-drive-downloader
    """

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = 'https://docs.google.com/uc?export=download'

    @staticmethod
    def download_file_from_google_drive(file_id, dest_path, overwrite=False, showsize=False):
        """
        Downloads a shared file from google drive into a given folder.
        Optionally unzips it.
        Parameters
        ----------
        file_id: str
            the file identifier.
            You can obtain it from the sharable link.
        dest_path: str
            the destination where to save the downloaded file.
            Must be a path (for example: './downloaded_file.txt')
        overwrite: bool
            optional, if True forces re-download and overwrite.
        showsize: bool
            optional, if True print the current download size.
        Returns
        -------
        None
        """

        destination_directory = os.path.dirname(dest_path)
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

        if not os.path.exists(dest_path) or overwrite:

            session = requests.Session()

            print('Downloading {} into {}... '.format(file_id, dest_path), end='')
            sys.stdout.flush()

            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL,
                                   params={'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL,
                                       params=params, stream=True)

            if showsize:
                print()  # Skip to the next line

            current_download_size = [0]
            GoogleDriveDownloader._save_response_content(response, dest_path, showsize,
                                                         current_download_size)
            print('Done.')

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination, showsize, current_size):
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    if showsize:
                        print('\r' + sizeof_fmt(current_size[0]), end=' ')
                        sys.stdout.flush()
                        current_size[0] += GoogleDriveDownloader.CHUNK_SIZE


def download(url: str,
             path: Optional[str] = None,
             overwrite: Optional[bool] = False,
             sha1_hash: Optional[str] = None,
             retries: Optional[int] = 5,
             verify_ssl: Optional[bool] = True) -> str:
    """Download a given URL

    Parameters
    ----------
    url
        URL to download
    path
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite
        Whether to overwrite destination file if already exists.
    sha1_hash
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl
        Verify SSL certificates.

    Returns
    -------
    fname
        The file path of the downloaded file.
    """
    from mxnet.gluon.utils import replace_file
    is_s3 = url.startswith(S3_PREFIX)
    if is_s3:
        boto3 = try_import_boto3()
        s3 = boto3.resource('s3')
        components = url[len(S3_PREFIX):].split('/')
        if len(components) < 2:
            raise ValueError('Invalid S3 url. Received url={}'.format(url))
        s3_bucket_name = components[0]
        s3_key = '/'.join(components[1:])
    if path is None:
        fname = url.split('/')[-1]
        # Empty filenames are invalid
        assert fname, 'Can\'t construct file-name from this URL. ' \
            'Please set the `path` option manually.'
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0, currently it's {}".format(
        retries)

    if not verify_ssl:
        warnings.warn(
            'Unverified HTTPS request is being made (verify_ssl=False). '
            'Adding certificate verification is strongly advised.')

    if overwrite or not os.path.exists(fname) or (sha1_hash and not sha1sum(fname) == sha1_hash):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print('Downloading {} from {}...'.format(fname, url))
                if is_s3:
                    response = s3.meta.client.head_object(Bucket=s3_bucket_name,
                                                          Key=s3_key)
                    total_size = int(response.get('ContentLength', 0))
                    random_uuid = str(uuid.uuid4())
                    tmp_path = '{}.{}'.format(fname, random_uuid)
                    if tqdm is not None:
                        def hook(t_obj):
                            def inner(bytes_amount):
                                t_obj.update(bytes_amount)
                            return inner
                        with tqdm.tqdm(total=total_size, unit='iB', unit_scale=True) as t:
                            s3.meta.client.download_file(s3_bucket_name, s3_key, tmp_path,
                                                         Callback=hook(t))
                    else:
                        s3.meta.client.download_file(s3_bucket_name, s3_key, tmp_path)
                else:
                    r = requests.get(url, stream=True, verify=verify_ssl)
                    if r.status_code != 200:
                        raise RuntimeError('Failed downloading url {}'.format(url))
                    # create uuid for temporary files
                    random_uuid = str(uuid.uuid4())
                    total_size = int(r.headers.get('content-length', 0))
                    chunk_size = 1024
                    if tqdm is not None:
                        t = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True)
                    with open('{}.{}'.format(fname, random_uuid), 'wb') as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:  # filter out keep-alive new chunks
                                if tqdm is not None:
                                    t.update(len(chunk))
                                f.write(chunk)
                    if tqdm is not None:
                        t.close()
                # if the target file exists(created by other processes)
                # and have the same hash with target file
                # delete the temporary file
                if not os.path.exists(fname) or (sha1_hash and not sha1sum(fname) == sha1_hash):
                    # atmoic operation in the same file system
                    replace_file('{}.{}'.format(fname, random_uuid), fname)
                else:
                    try:
                        os.remove('{}.{}'.format(fname, random_uuid))
                    except OSError:
                        pass
                    finally:
                        warnings.warn(
                            'File {} exists in file system so the downloaded file is deleted'.format(fname))
                if sha1_hash and not sha1sum(fname) == sha1_hash:
                    raise UserWarning(
                        'File {} is downloaded but the content hash does not match.'
                        ' The repo may be outdated or download may be incomplete. '
                        'If the "repo_url" is overridden, consider switching to '
                        'the default repo.'.format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e

                print('download failed due to {}, retrying, {} attempt{} left'
                      .format(repr(e), retries, 's' if retries > 1 else ''))

    return fname


def check_version(min_version: str,
                  warning_only: bool = False,
                  library: Optional[ModuleType] = None):
    """Check the version of gluonnlp satisfies the provided minimum version.
    An exception is thrown if the check does not pass.

    Parameters
    ----------
    min_version
        Minimum version
    warning_only
        Printing a warning instead of throwing an exception.
    library
        The target library for version check. Checks gluonnlp by default
    """
    # pylint: disable=import-outside-toplevel
    from .. import __version__
    if library is None:
        version = __version__
        name = 'GluonNLP'
    else:
        version = library.__version__
        name = library.__name__
    from packaging.version import parse
    bad_version = parse(version.replace('.dev', '')) < parse(min_version)
    if bad_version:
        msg = 'Installed {} version {} does not satisfy the ' \
              'minimum required version {}'.format(name, version, min_version)
        if warning_only:
            warnings.warn(msg)
        else:
            raise AssertionError(msg)


def num_mp_workers(max_worker=4):
    """Get the default number multiprocessing workers

    Parameters
    ----------
    max_worker
        The max allowed number of workers

    Returns
    -------
    num_max_worker
        number of max worker
    """
    import multiprocessing as mp
    return min(mp.cpu_count(), max_worker)


def get_mxnet_visible_gpus():
    """Get the number of GPUs that are visible to MXNet.

    Returns
    -------
    ctx_l
        The ctx list
    """
    import mxnet as mx
    gpu_count = 0
    while True:
        try:
            arr = mx.np.array(1.0, ctx=mx.gpu(gpu_count))
            arr.asnumpy()
            gpu_count += 1
        except Exception:
            break
    return [mx.gpu(i) for i in range(gpu_count)]


def get_mxnet_available_ctx():
    """

    Returns
    -------
    ctx_l
        Get the available contexts
    """
    import mxnet as mx
    gpu_ctx_l = get_mxnet_visible_gpus()
    if len(gpu_ctx_l) == 0:
        ctx_l = [mx.cpu()]
    else:
        ctx_l = gpu_ctx_l
    return ctx_l
