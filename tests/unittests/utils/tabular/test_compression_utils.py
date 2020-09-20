import gzip
import bz2
import lzma

from autogluon.utils.tabular.utils import compression_utils


def test_get_validated_path_no_compression_fn():
    no_extension_filepath = 'dummy_file'
    assert(compression_utils.get_validated_path(no_extension_filepath) == no_extension_filepath)

    single_extension_filepath = 'dummy_file.pkl'
    assert(compression_utils.get_validated_path(single_extension_filepath) == single_extension_filepath)

    multiple_extension_filepath = 'dummy_file.fake-foo.zip.pkl'
    assert(compression_utils.get_validated_path(multiple_extension_filepath) == multiple_extension_filepath)


def test_get_validated_path_with_compression_fn():
    compression_fns = ['gzip', 'lzma', 'bz2']

    expected_gzip_filepath = "dummy_file.gz"
    expected_lzma_filepath = "dummy_file.lzma"
    expected_bz2_filepath = "dummy_file.bz2"

    no_extension_filepath = 'dummy_file'
    assert(compression_utils.get_validated_path(no_extension_filepath, compression_fns[0])
           == expected_gzip_filepath)
    assert(compression_utils.get_validated_path(no_extension_filepath, compression_fns[1])
           == expected_lzma_filepath)
    assert(compression_utils.get_validated_path(no_extension_filepath, compression_fns[2])
           == expected_bz2_filepath)

    single_extension_filepath = 'dummy_file.pkl'
    assert(compression_utils.get_validated_path(single_extension_filepath, compression_fns[0])
           == expected_gzip_filepath)
    assert(compression_utils.get_validated_path(single_extension_filepath, compression_fns[1])
           == expected_lzma_filepath)
    assert(compression_utils.get_validated_path(single_extension_filepath, compression_fns[2])
           == expected_bz2_filepath)

    multiple_extension_filepath = 'dummy_file.fake-foo.zip.pkl'
    assert (compression_utils.get_validated_path(multiple_extension_filepath, compression_fns[0])
            == expected_gzip_filepath)
    assert (compression_utils.get_validated_path(multiple_extension_filepath, compression_fns[1])
            == expected_lzma_filepath)
    assert (compression_utils.get_validated_path(multiple_extension_filepath, compression_fns[2])
            == expected_bz2_filepath)


def test_get_compression_map():
    expected_compression_fn_map = {
        None: {
            'open': open,
            'extension': 'pkl',
        },
        'gzip': {
            'open': gzip.open,
            'extension': 'gz',
        },
        'bz2': {
            'open': bz2.open,
            'extension': 'bz2',
        },
        'lzma': {
            'open': lzma.open,
            'extension': 'lzma',
        },
    }
    assert compression_utils.get_compression_map() == expected_compression_fn_map
