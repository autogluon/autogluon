class FitContextMemo:
    """Context limits shared by model copies that belong to one AutoGluon fit."""

    def __init__(self):
        self.values: dict[tuple, tuple[int, int]] = {}

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self
