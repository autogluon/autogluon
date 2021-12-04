import autogluon.common


def test_import_version():
    assert isinstance(autogluon.common.__version__, str)
    assert len(autogluon.common.__version__) != 0
