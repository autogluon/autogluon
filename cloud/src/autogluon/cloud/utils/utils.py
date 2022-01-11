import uuid
import logging

logger = logging.getLogger(__name__)


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
