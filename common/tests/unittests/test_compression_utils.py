from autogluon.common.utils import compression_utils


def test_get_validated_path_no_compression_fn():
    no_extension_filepath = "dummy_file"
    assert compression_utils.get_validated_path(no_extension_filepath) == no_extension_filepath

    single_extension_filepath = "dummy_file.pkl"
    assert compression_utils.get_validated_path(single_extension_filepath) == single_extension_filepath

    multiple_extension_filepath = "dummy_file.fake-foo.zip.pkl"
    assert compression_utils.get_validated_path(multiple_extension_filepath) == multiple_extension_filepath


def test_get_validated_path_with_compression_fn():
    compression_fns = ["gzip", "lzma", "bz2"]

    no_extension_filepath = "dummy_file"

    expected_no_ext_gzip_filepath = "dummy_file.gz"
    expected_no_ext_lzma_filepath = "dummy_file.lzma"
    expected_no_ext_bz2_filepath = "dummy_file.bz2"
    assert (
        compression_utils.get_validated_path(no_extension_filepath, compression_fns[0])
        == expected_no_ext_gzip_filepath
    )
    assert (
        compression_utils.get_validated_path(no_extension_filepath, compression_fns[1])
        == expected_no_ext_lzma_filepath
    )
    assert (
        compression_utils.get_validated_path(no_extension_filepath, compression_fns[2]) == expected_no_ext_bz2_filepath
    )

    single_extension_filepath = "dummy_file.pkl"

    expected_pkl_gzip_filepath = "dummy_file.pkl.gz"
    expected_pkl_lzma_filepath = "dummy_file.pkl.lzma"
    expected_pkl_bz2_filepath = "dummy_file.pkl.bz2"
    assert (
        compression_utils.get_validated_path(single_extension_filepath, compression_fns[0])
        == expected_pkl_gzip_filepath
    )
    assert (
        compression_utils.get_validated_path(single_extension_filepath, compression_fns[1])
        == expected_pkl_lzma_filepath
    )
    assert (
        compression_utils.get_validated_path(single_extension_filepath, compression_fns[2])
        == expected_pkl_bz2_filepath
    )

    multiple_extension_filepath = "dummy_file.fake-foo.zip.pkl"

    expected_multi_gzip_filepath = "dummy_file.fake-foo.zip.pkl.gz"
    expected_multi_lzma_filepath = "dummy_file.fake-foo.zip.pkl.lzma"
    expected_multi_bz2_filepath = "dummy_file.fake-foo.zip.pkl.bz2"
    assert (
        compression_utils.get_validated_path(multiple_extension_filepath, compression_fns[0])
        == expected_multi_gzip_filepath
    )
    assert (
        compression_utils.get_validated_path(multiple_extension_filepath, compression_fns[1])
        == expected_multi_lzma_filepath
    )
    assert (
        compression_utils.get_validated_path(multiple_extension_filepath, compression_fns[2])
        == expected_multi_bz2_filepath
    )


def test_get_compression_map():
    expected_compression_fn_map = {
        None: {
            "open": open,
            "extension": "",
        },
        "gzip": {
            "open": compression_utils._gzip_open,
            "extension": "gz",
        },
        "bz2": {
            "open": compression_utils._bz2_open,
            "extension": "bz2",
        },
        "lzma": {
            "open": compression_utils._lzma_open,
            "extension": "lzma",
        },
    }
    assert compression_utils.get_compression_map() == expected_compression_fn_map
