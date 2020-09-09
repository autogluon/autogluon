"""Model store which provides pretrained models."""
from __future__ import print_function

__all__ = ['get_model_file', 'purge']

import os
import zipfile

from ..utils import download, check_sha1


_model_sha1 = {name: checksum for checksum, name in [
    ('dd74519f956bc608912e413316a7eaa7fac4b365', 'efficientnet_b0'),
    ('d920bff669b8c0cadd3107b984d8a52748d341a4', 'efficientnet_b1'),
    ('3a552326d08c80cecbe1abf6f209223dfcaf7b30', 'efficientnet_b2'),
    ('8ed810a38e73ec71659b110919e59bcb4fbebd8f', 'efficientnet_b3'),
    ('20319a29bb6cd3c0bb0c4d4dce9f0d5a279f7b7e', 'efficientnet_b4'),
    ('e0da163506c2aa3ad48b92829af75162e09e0b7b', 'efficientnet_b5'),
    ('eb97a9dab456c9931673e469e0a09a0d3fda2a11', 'efficientnet_b6'),
    ('3cb5cc71e074ddb8eb1b334d1099fc1ecaa78562', 'efficientnet_b7'),
    ('96443327d8113ae5b4346db1cd29b96b361eed72', 'standford_dog_resnet152_v1'),
    ('5dbc5d6789b0f6fd8c27974e7bb68db540c2e2e5', 'standford_dog_resnext101_64x4d'),
    ]}

autogluon_repo_url = 'https://autogluon.s3.amazonaws.com/'
_url_format = '{repo_url}models/{file_name}.zip'

def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]

def get_model_file(name, root=os.path.join('~', '.autogluon', 'models')):
    r"""Return location for the pretrained on local file system.
    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.
    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    file_name = '{name}-{short_hash}'.format(name=name, short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name+'.params')
    sha1_hash = _model_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            print('Mismatch in the content of model file {} detected.' +
                  ' Downloading again.'.format(file_path))
    else:
        print('Model file {} is not found. Downloading.'.format(file_path))

    if not os.path.exists(root):
        os.makedirs(root)

    zip_file_path = os.path.join(root, file_name+'.zip')
    repo_url = os.environ.get('ENCODING_REPO', autogluon_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=zip_file_path,
             overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)
    os.remove(zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')

def purge(root=os.path.join('~', '.encoding', 'models')):
    r"""Purge all pretrained model files in local file store.
    Parameters
    ----------
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".params"):
            os.remove(os.path.join(root, f))

def pretrained_model_list():
    return list(_model_sha1.keys())
