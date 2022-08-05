import logging
import os
import tarfile
import shutil
import uuid
import zipfile


logger = logging.getLogger(__name__)


def zipfolder(output_filename, dir_name):
    """
    Zip a folder while preserving the directory structure

    Example
        If dir_name is temp, and the structure is as follow:
        home
        |--- temp
             |---train
             |---test
        The zipped file will also has temp as the base directory instead of train and test
    """
    dir_name = os.path.abspath(dir_name)
    root_dir = os.path.dirname(dir_name)
    base_dir = os.path.basename(dir_name)
    shutil.make_archive(output_filename, 'zip', root_dir, base_dir)


def is_compressed_file(filename):
    return tarfile.is_tarfile(filename) or zipfile.is_zipfile(filename)


def unzip_file(tarball_path, save_path):
    file = tarfile.open(tarball_path)
    file.extractall(save_path)
    file.close()


def rename_file_with_uuid(file_name):
    tmp = file_name.rsplit('.', 1)
    name = tmp[0]
    if len(tmp) > 1:
        extension = tmp[1]
        joiner = '.'
    else:
        extension = ''
        joiner = ''
    new_file_name = name + '_' + str(uuid.uuid4()) + joiner + extension
    return new_file_name
