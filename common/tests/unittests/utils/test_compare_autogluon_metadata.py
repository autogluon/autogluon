import copy
import unittest

from autogluon.common.utils.utils import compare_autogluon_metadata, get_autogluon_metadata


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


if __name__ == "__main__":
    unittest.main()
