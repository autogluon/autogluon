import warnings
from warnings import warn

__all__ = ['AutoGluonWarning', 'make_deprecate', 'DeprecationHelper']

class AutoGluonWarning(DeprecationWarning):
    pass

warnings.simplefilter('once', AutoGluonWarning)

def make_deprecate(meth, old_name):
    """TODO Add Docs
    """
    new_name = meth.__name__
    def deprecated_init(*args, **kwargs):
        warn("autogluon.{} is now deprecated in favor of autogluon.{}."
             .format(old_name, new_name), AutoGluonWarning)
        return meth(*args, **kwargs)

    deprecated_init.__doc__ = r"""
    {old_name}(...)
    .. warning::
        This method is now deprecated in favor of :func:`autogluon.{new_name}`. \
    See :func:`~autogluon.{new_name}` for details.""".format(
        old_name=old_name, new_name=new_name)
    deprecated_init.__name__ = old_name
    return deprecated_init


class DeprecationHelper(object):
    """TODO Add Docs
    """
    def __init__(self, old_class, new_name):
        self.old_class = old_class
        self.old_name = old_class.__name__
        self.new_name = new_name

    def _warn(self):
        warn("autogluon.{} is now deprecated in favor of autogluon.{}." \
             .format(self.old_name, self.new_name), AutoGluonWarning)

    def __call__(self, *args, **kwargs):
        self._warn()
        return self.old_class(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.old_class, attr)
