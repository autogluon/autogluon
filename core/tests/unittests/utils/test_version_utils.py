from autogluon.core.utils import show_versions


def test_show_versions():
    # Only verify this function does not crash
    show_versions()
