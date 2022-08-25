def post_fit(func):
    """decorator for post-fit methods"""

    def pff_wrapper(self, *args, **kwargs):
        assert self._is_fit, f'.fit needs to be called prior to .{func.__name__}'
        return func(self, *args, **kwargs)

    return pff_wrapper