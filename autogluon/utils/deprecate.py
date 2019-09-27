def _make_deprecate(meth, old_name):
    new_name = meth.__name__

    def deprecated_init(*args, **kwargs):
        warnings.warn("autogluon.{} is now deprecated in favor of autogluon.{}."
                      .format(old_name, new_name), EncodingDeprecationWarning)
        return meth(*args, **kwargs)

    deprecated_init.__doc__ = r"""
    {old_name}(...)
    .. warning::
        This method is now deprecated in favor of :func:`autogluon.{new_name}`.
    See :func:`~autogluon.{new_name}` for details.""".format(
        old_name=old_name, new_name=new_name)
    deprecated_init.__name__ = old_name
    return deprecated_init
