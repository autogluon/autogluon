class AnalysisState(dict):
    """Enabling dot.notation access to dictionary attributes and dynamic code assist in jupyter"""
    __getattr__ = dict.get
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs) -> None:
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        for k, v in kwargs.items():
            self[k] = v

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, dict):
            value = AnalysisState(value)
        self[name] = value

    def __setitem__(self, key, value) -> None:
        if isinstance(value, dict):
            value = AnalysisState(value)
        super().__setitem__(key, value)

    @property
    def __dict__(self):
        return self
