import autogluon.core


def test_import_version():
    assert isinstance(autogluon.core.__version__, str)
    assert len(autogluon.core.__version__) != 0
