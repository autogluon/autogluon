import copy
import unittest

from autogluon.common.utils.utils import (
    check_saved_predictor_version,
    compare_autogluon_metadata,
    get_autogluon_metadata,
)


class CompareAutoGluonMetadataTestCase(unittest.TestCase):
    def test_no_warnings(self):
        metadata = get_autogluon_metadata()
        logs = compare_autogluon_metadata(original=metadata, current=metadata)
        assert len(logs) == 0

    def test_version_mismatch(self):
        metadata_og = get_autogluon_metadata()
        metadata_cu = copy.deepcopy(metadata_og)
        metadata_cu["version"] = "dummy_version"
        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 1

    def test_py_version_mismatch(self):
        metadata_og = get_autogluon_metadata()
        metadata_cu = copy.deepcopy(metadata_og)
        metadata_cu["py_version"] = "dummy_py_version"
        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 1

    def test_py_version_micro_mismatch(self):
        metadata_og = get_autogluon_metadata()
        metadata_cu = copy.deepcopy(metadata_og)
        metadata_cu["py_version_micro"] = "dummy_py_version_micro"
        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 1

    def test_py_version_both_mismatch(self):
        metadata_og = get_autogluon_metadata()
        metadata_cu = copy.deepcopy(metadata_og)
        metadata_cu["py_version"] = "dummy_py_version"
        metadata_cu["py_version_micro"] = "dummy_py_version_micro"
        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 1

    def test_system_mismatch(self):
        metadata_og = get_autogluon_metadata()
        metadata_cu = copy.deepcopy(metadata_og)
        metadata_cu["system"] = "dummy_system"
        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 1

    def test_combined_mismatch(self):
        metadata_og = get_autogluon_metadata()
        metadata_cu = copy.deepcopy(metadata_og)
        metadata_cu["version"] = "dummy_version"
        metadata_cu["py_version"] = "dummy_py_version"
        metadata_cu["system"] = "dummy_system"
        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 3

    def test_new_key(self):
        metadata_og = get_autogluon_metadata()
        metadata_cu = copy.deepcopy(metadata_og)
        metadata_cu["new_key"] = "dummy_val"
        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 0

    def test_package_mismatch(self):
        metadata_og = get_autogluon_metadata()
        metadata_cu = copy.deepcopy(metadata_og)
        metadata_og["packages"]["dummy_package"] = "0.2"

        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 1
        assert logs[0] == (30, "WARNING: Missing package 'dummy_package==0.2'")

        metadata_cu["packages"]["dummy_package"] = "0.3"
        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 1
        assert logs[0] == (30, "WARNING: Package version diff 'dummy_package'\t(original=0.2, current=0.3)")

        del metadata_og["packages"]["dummy_package"]
        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 1
        assert logs[0] == (30, "INFO: New package 'dummy_package==0.3'")


class CheckSavedPredictorVersionTestCase(unittest.TestCase):
    def test_exact_match_does_not_raise(self):
        check_saved_predictor_version("1.2.0", "1.2.0", require_version_match=True)

    def test_patch_version_mismatch_does_not_raise(self):
        check_saved_predictor_version("1.2.1", "1.2.0", require_version_match=True)

    def test_patch_version_mismatch_reverse_does_not_raise(self):
        check_saved_predictor_version("1.2.0", "1.2.1", require_version_match=True)

    def test_minor_version_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            check_saved_predictor_version("1.3.0", "1.2.0", require_version_match=True)

    def test_major_version_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            check_saved_predictor_version("2.0.0", "1.2.0", require_version_match=True)

    def test_unparseable_version_raises(self):
        with self.assertRaises(AssertionError):
            check_saved_predictor_version("1.2.0", "Unknown (Likely <=0.7.0)", require_version_match=True)

    def test_prerelease_version_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            check_saved_predictor_version("1.2.0", "1.2.0a1", require_version_match=True)

    def test_minor_version_mismatch_no_raise_when_require_false(self):
        check_saved_predictor_version("1.3.0", "1.2.0", require_version_match=False)

    def test_patch_version_mismatch_metadata_is_info(self):
        metadata_og = get_autogluon_metadata()
        metadata_cu = copy.deepcopy(metadata_og)
        metadata_og["version"] = "1.2.0"
        metadata_cu["version"] = "1.2.1"
        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 1
        assert logs[0][0] == 20  # INFO level

    def test_minor_version_mismatch_metadata_is_warning(self):
        metadata_og = get_autogluon_metadata()
        metadata_cu = copy.deepcopy(metadata_og)
        metadata_og["version"] = "1.2.0"
        metadata_cu["version"] = "1.3.0"
        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 1
        assert logs[0][0] == 30  # WARNING level

    def test_unparseable_version_metadata_is_warning(self):
        metadata_og = get_autogluon_metadata()
        metadata_cu = copy.deepcopy(metadata_og)
        metadata_og["version"] = "1.2.0"
        metadata_cu["version"] = "Unknown (Likely <=0.7.0)"
        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 1
        assert logs[0][0] == 30  # WARNING level

    def test_prerelease_version_metadata_is_warning(self):
        metadata_og = get_autogluon_metadata()
        metadata_cu = copy.deepcopy(metadata_og)
        metadata_og["version"] = "1.2.0"
        metadata_cu["version"] = "1.2.0a1"
        logs = compare_autogluon_metadata(original=metadata_og, current=metadata_cu)
        assert len(logs) == 1
        assert logs[0][0] == 30  # WARNING level


if __name__ == "__main__":
    unittest.main()
