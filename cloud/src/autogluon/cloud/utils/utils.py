import uuid
import logging

logger = logging.getLogger(__name__)


def rename_file_with_uuid(file_name):
    name, extension = file_name.split('.')[0], file_name.split('.')[1]
    new_name = name + '_' + str(uuid.uuid4())
    return new_name + extension
