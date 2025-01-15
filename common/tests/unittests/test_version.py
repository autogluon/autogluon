from packaging import version

from autogluon.common import __version__


def test_version_has_major_minor_micro():
    """
    Verifies that the version contains a major minor and micro version explicitly.

    For example, `__version__ = "1.2"` will fail this test because it does not contain an explicit micro version.
    """
    v = version.parse(__version__)

    major = v.major
    minor = v.minor
    micro = v.micro

    assert __version__.startswith(f"{major}.{minor}.{micro}")
