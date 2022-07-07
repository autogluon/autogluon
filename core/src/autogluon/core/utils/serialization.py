"""Serilization and checkpoint"""
import logging
import difflib
import inspect
import os
import io
import struct
import sys
import tarfile
import zipfile
import warnings
from contextlib import closing

if sys.version_info[0] == 2:
    import cPickle as pickle
    _string_classes = basestring
else:
    import pickle
    import pathlib
    _string_classes = (str, bytes)

__all__ = ['save', 'load']

DEFAULT_PROTOCOL = 2

LONG_SIZE = struct.Struct('=l').size
INT_SIZE = struct.Struct('=i').size
SHORT_SIZE = struct.Struct('=h').size

logger = logging.getLogger(__name__)
MAGIC_NUMBER = 0x7df059597099bb7dcf25
PROTOCOL_VERSION = 1001
STORAGE_KEY_SEPARATOR = ','

_package_registry = []

def save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL):
    """Saves an object to local file.

    Parameters
    ----------
    obj : object
        Python object to save.
    f : string or file object
        A file-like object (has to implement write and flush), or a string containing a file name
    pickle_module : pickle
        Module used for pickling metadata and objects
    pickle_protocol : protocol (optional)
        Protocol can be specified to override the default pickle protocol.

    Examples
    --------
    >>> save(scheduler.state_dict(), checkname)
    """
    return _with_file_like(f, "wb", lambda f: _save(obj, f, pickle_module, pickle_protocol))

def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    """Loads an object saved with :func:`save` from file.

    Parameters
    ----------
        f (string or file object): a file-like object (has to implement write and flush)
            or a string containing a file name

    Examples
    --------
    >>> scheduler.load_state_dict(load(checkpoint))
    """
    new_fd = False
    if isinstance(f, str) or \
            (sys.version_info[0] == 2 and isinstance(f, unicode)):
        new_fd = True
        f = open(f, 'rb')
    elif (sys.version_info[0] == 3 and isinstance(f, pathlib.Path)):
        new_fd = True
        f = f.open('rb')
    try:
        return _load(f, map_location, pickle_module, **pickle_load_args)
    finally:
        if new_fd:
            f.close()

def _with_file_like(f, mode, body):
    """
    Executes a body function with a file object for f, opening
    it in 'mode' if it is a string filename.
    """
    new_fd = False
    if isinstance(f, str) or \
            (sys.version_info[0] == 2 and isinstance(f, unicode)) or \
            (sys.version_info[0] == 3 and isinstance(f, pathlib.Path)):
        new_fd = True
        f = open(f, mode)
    try:
        return body(f)
    finally:
        if new_fd:
            f.close()


def register_package(priority, tagger, deserializer):
    queue_elem = (priority, tagger, deserializer)
    _package_registry.append(queue_elem)
    _package_registry.sort()

def _cpu_deserialize(obj, location):
    if location == 'cpu':
        return obj

def _cpu_tag(obj):
    if type(obj).__module__ == 'mxnet':
        return 'cpu'

register_package(10, _cpu_tag, _cpu_deserialize)

def default_restore_location(storage, location):
    for _, _, fn in _package_registry:
        result = fn(storage, location)
        if result is not None:
            return result
    raise RuntimeError("don't know how to restore data location")

def _is_compressed_file(f):
    compress_modules = ['gzip']
    try:
        return f.__module__ in compress_modules
    except AttributeError:
        return False

def _should_read_directly(f):
    """
    Checks if f is a file that should be read directly. It should be read
    directly if it is backed by a real file (has a fileno) and is not a
    a compressed file (e.g. gzip)
    """
    if _is_compressed_file(f):
        return False
    try:
        return f.fileno() >= 0
    except io.UnsupportedOperation:
        return False
    except AttributeError:
        return False

def _check_seekable(f):
    def raise_err_msg(patterns, e):
        for p in patterns:
            if p in str(e):
                msg = (str(e) + ". You can only load from a file that is seekable." +
                                " Please pre-load the data into a buffer like io.BytesIO and" +
                                " try to load from it instead.")
                raise type(e)(msg)
        raise e

    try:
        f.seek(f.tell())
        return True
    except (io.UnsupportedOperation, AttributeError) as e:
        raise_err_msg(["seek", "tell"], e)


def _save(obj, f, pickle_module, pickle_protocol):
    if sys.version_info[0] == 2:
        import StringIO
        if isinstance(f, StringIO.StringIO):
            msg = ('save received unsupported StringIO.StringIO file object, whose '
                   'write method does not return the number of bytes written. '
                   'Please use something like io.BytesIO for save instead.')
            logger.error(msg)
            raise RuntimeError(msg)

    serialized_container_types = {}
    serialized_storages = {}

    sys_info = dict(
        protocol_version=PROTOCOL_VERSION,
        little_endian=sys.byteorder == 'little',
        type_sizes=dict(
            short=SHORT_SIZE,
            int=INT_SIZE,
            long=LONG_SIZE,
        ),
    )

    pickle_module.dump(MAGIC_NUMBER, f, protocol=pickle_protocol)
    pickle_module.dump(PROTOCOL_VERSION, f, protocol=pickle_protocol)
    pickle_module.dump(sys_info, f, protocol=pickle_protocol)
    pickler = pickle_module.Pickler(f, protocol=pickle_protocol)
    pickler.dump(obj)

    serialized_storage_keys = sorted(serialized_storages.keys())
    pickle_module.dump(serialized_storage_keys, f, protocol=pickle_protocol)
    f.flush()
    for key in serialized_storage_keys:
        serialized_storages[key]._write_file(f, _should_read_directly(f))


def _load(f, map_location, pickle_module, **pickle_load_args):
    deserialized_objects = {}

    if map_location is None:
        restore_location = default_restore_location
    elif isinstance(map_location, dict):
        def restore_location(storage, location):
            location = map_location.get(location, location)
            return default_restore_location(storage, location)
    elif isinstance(map_location, _string_classes):
        def restore_location(storage, location):
            return default_restore_location(storage, map_location)
    else:
        def restore_location(storage, location):
            result = map_location(storage, location)
            if result is None:
                result = default_restore_location(storage, location)
            return result

    def _check_container_source(container_type, source_file, original_source):
        try:
            current_source = inspect.getsource(container_type)
        except Exception:  # saving the source is optional, so we can ignore any errors
            warnings.warn("Couldn't retrieve source code for container of "
                          "type " + container_type.__name__ + ". It won't be checked "
                          "for correctness upon loading.")
            return
        if original_source != current_source:
            if container_type.dump_patches:
                file_name = container_type.__name__ + '.patch'
                diff = difflib.unified_diff(current_source.split('\n'),
                                            original_source.split('\n'),
                                            source_file,
                                            source_file, lineterm="")
                lines = '\n'.join(diff)
                try:
                    with open(file_name, 'a+') as f:
                        file_size = f.seek(0, 2)
                        f.seek(0)
                        if file_size == 0:
                            f.write(lines)
                        elif file_size != len(lines) or f.read() != lines:
                            raise IOError
                    msg = ("Saved a reverse patch to " + file_name + ". "
                           "Run `patch -p0 < " + file_name + "` to revert your "
                           "changes.")
                except IOError:
                    msg = ("Tried to save a patch, but couldn't create a "
                           "writable file " + file_name + ". Make sure it "
                           "doesn't exist and your working directory is "
                           "writable.")
            else:
                msg = ("you can retrieve the original source code by "
                       "accessing the object's source attribute")
            msg = ("source code of class has changed. {}"
                   .format(msg))
            warnings.warn(msg, SourceChangeWarning)

    deserialized_objects = {}

    def maybe_decode_ascii(bytes_str):
        # When using encoding='bytes' in Py3, some **internal** keys stored as
        # strings in Py2 are loaded as bytes. This function decodes them with
        # ascii encoding, one that Py3 uses by default.
        #
        # NOTE: This should only be used on internal keys (e.g., `typename` and
        #       `location` in `persistent_load` below!
        if isinstance(bytes_str, bytes):
            return bytes_str.decode('ascii')
        return bytes_str

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        if typename == 'module':
            # Ignore containers that don't have any sources saved
            if all(data[1:]):
                _check_container_source(*data)
            return data[0]
        elif typename == 'storage':
            data_type, root_key, location, size, view_metadata = data
            location = maybe_decode_ascii(location)
            if root_key not in deserialized_objects:
                obj = data_type(size)
                deserialized_objects[root_key] = restore_location(obj, location)
            storage = deserialized_objects[root_key]
            if view_metadata is not None:
                view_key, offset, view_size = view_metadata
                if view_key not in deserialized_objects:
                    deserialized_objects[view_key] = storage[offset:offset + view_size]
                return deserialized_objects[view_key]
            else:
                return storage
        else:
            raise RuntimeError("Unknown saved id type: %s" % saved_id[0])

    def legacy_load(f):
        deserialized_objects = {}

        def persistent_load(saved_id):
            if isinstance(saved_id, tuple):
                # Ignore containers that don't have any sources saved
                if all(saved_id[1:]):
                    _check_container_source(*saved_id)
                return saved_id[0]
            return deserialized_objects[int(saved_id)]

        with closing(tarfile.open(fileobj=f, mode='r:', format=tarfile.PAX_FORMAT)) as tar, \
                mkdtemp() as tmpdir:

            tar.extract('storages', path=tmpdir)
            with open(os.path.join(tmpdir, 'storages'), 'rb', 0) as f:
                num_storages = pickle_module.load(f, **pickle_load_args)
                for i in range(num_storages):
                    args = pickle_module.load(f, **pickle_load_args)
                    key, location, storage_type = args
                    obj = storage_type._new_with_file(f)
                    obj = restore_location(obj, location)
                    deserialized_objects[key] = obj

                storage_views = pickle_module.load(f, **pickle_load_args)
                for target_cdata, root_cdata, offset, size in storage_views:
                    root = deserialized_objects[root_cdata]
                    deserialized_objects[target_cdata] = root[offset:offset + size]

            tar.extract('tensors', path=tmpdir)
            with open(os.path.join(tmpdir, 'tensors'), 'rb', 0) as f:
                num_tensors = pickle_module.load(f, **pickle_load_args)
                for _ in range(num_tensors):
                    args = pickle_module.load(f, **pickle_load_args)
                    key, storage_id, original_tensor_type = args
                    storage = deserialized_objects[storage_id]
                    tensor_type = storage_to_tensor_type(storage)
                    ndim, = struct.unpack('<i', f.read(4))
                    # skip next 4 bytes; legacy encoding treated ndim as 8 bytes
                    f.read(4)
                    size = struct.unpack('<{}q'.format(ndim), f.read(8 * ndim))
                    stride = struct.unpack('<{}q'.format(ndim), f.read(8 * ndim))
                    storage_offset, = struct.unpack('<q', f.read(8))
                    tensor = tensor_type().set_(storage, storage_offset, size, stride)
                    deserialized_objects[key] = tensor

            pickle_file = tar.extractfile('pickle')
            unpickler = pickle_module.Unpickler(pickle_file, **pickle_load_args)
            unpickler.persistent_load = persistent_load
            result = unpickler.load()
            return result

    _check_seekable(f)
    f_should_read_directly = _should_read_directly(f)

    if f_should_read_directly and f.tell() == 0:
        try:
            return legacy_load(f)
        except tarfile.TarError:
            if zipfile.is_zipfile(f):
                raise RuntimeError("Please uncompress the file.")
            # if not a tarfile, reset file offset and proceed
            f.seek(0)

    magic_number = pickle_module.load(f, **pickle_load_args)
    if magic_number != MAGIC_NUMBER:
        raise RuntimeError("Invalid magic number; corrupt file?")
    protocol_version = pickle_module.load(f, **pickle_load_args)
    if protocol_version != PROTOCOL_VERSION:
        raise RuntimeError("Invalid protocol version: %s" % protocol_version)

    _sys_info = pickle_module.load(f, **pickle_load_args)
    unpickler = pickle_module.Unpickler(f, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    deserialized_storage_keys = pickle_module.load(f, **pickle_load_args)

    offset = f.tell() if f_should_read_directly else None
    for key in deserialized_storage_keys:
        assert key in deserialized_objects
        deserialized_objects[key]._set_from_file(f, offset, f_should_read_directly)
        offset = None

    return result
