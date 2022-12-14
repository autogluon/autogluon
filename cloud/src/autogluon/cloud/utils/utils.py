import base64
import copy
import logging
import os
import shutil
import tarfile
import uuid
import zipfile

import PIL
from PIL import Image

logger = logging.getLogger(__name__)


def get_real_image_path_in_image_column(image_root_path, path):
    """
    Combine root path and path, where path contains root path's basename.
    For example,
        image_root_path = /home/user/example_images/
        path = example_images/test/a.jpg
        would return /home/user/example_images/test/a.jpg
    This is needed to make sure the fit and predict api of CloudPredictor follows the same logic.
    """
    image_root_path = os.path.abspath(image_root_path)
    path = path.split(os.path.sep)
    while path[0] == '':
        # Avoid cases like /foo/test
        path.pop(0)
    image_path_root_head, image_path_tail = os.path.split(image_root_path)
    assert image_path_tail == path[0], 'Please make sure the image path inside your image column contains the root image directory'

    return os.path.join(image_path_root_head, *path)


def read_image_bytes_and_encode(image_path):
    image_obj = open(image_path, "rb")
    image_bytes = image_obj.read()
    image_obj.close()
    b85_image = base64.b85encode(image_bytes).decode("utf-8")

    return b85_image


def convert_image_path_to_encoded_bytes_in_dataframe(dataframe, image_column):
    assert image_column in dataframe, "Please specify a valid image column name"
    dataframe = copy.deepcopy(dataframe)
    dataframe[image_column] = [read_image_bytes_and_encode(path) for path in dataframe[image_column]]

    return dataframe


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
    shutil.make_archive(output_filename, "zip", root_dir, base_dir)


def is_compressed_file(filename):
    return tarfile.is_tarfile(filename) or zipfile.is_zipfile(filename)


def is_image_file(filename):
    try:
        Image.open(filename)
    except PIL.UnidentifiedImageError:
        # Not an image
        return False
    return True


def unzip_file(tarball_path, save_path):
    file = tarfile.open(tarball_path)
    file.extractall(save_path)
    file.close()


def rename_file_with_uuid(file_name):
    tmp = file_name.rsplit(".", 1)
    name = tmp[0]
    if len(tmp) > 1:
        extension = tmp[1]
        joiner = "."
    else:
        extension = ""
        joiner = ""
    new_file_name = name + "_" + str(uuid.uuid4()) + joiner + extension
    return new_file_name
