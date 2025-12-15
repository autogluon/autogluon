from __future__ import annotations


class Operation:
    def __init__(self, left, right, op: str):
        self.left = left
        self.right = right
        self.op = op

    def _render(self, x):
        if isinstance(x, Operation):
            return f"({x.name()})"
        return str(x)

    def name(self) -> str:
        left = self._render(self.left)
        right = self._render(self.right)
        return f"{left}{self.op}{right}"

    def __str__(self):
        return self.name()
