from autogluon.common.utils.version_utils import VersionManager
from autogluon.common import __version__


def test_version_manager():
    assert VersionManager.get_ag_version("common") == __version__


def test_mock_version(mock_ag_version_mgr):
    real_version = VersionManager.get_ag_version("common")
    mock_version = "9999.0"
    with mock_ag_version_mgr(mock_version):
        assert VersionManager.get_ag_version("common") == mock_version
    assert VersionManager.get_ag_version("common") == real_version
