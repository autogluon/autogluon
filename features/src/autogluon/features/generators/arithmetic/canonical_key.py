from __future__ import annotations

from typing import Dict, Hashable, List, Tuple, Union

import numpy as np

from .operation import Operation

# ----------------------------
# Internal polynomial utilities
# ----------------------------

# A monomial is represented by a frozenset of (var, exponent) pairs.
Monom = frozenset[tuple[str, int]]
Poly = Dict[Monom, int]  # monom -> integer coefficient


def _monom_from_var(var: str, exp: int = 1) -> Monom:
    if exp == 0:
        return frozenset()
    return frozenset(((var, exp),))


def _monom_add(m: Monom, var: str, delta: int) -> Monom:
    d = dict(m)
    d[var] = d.get(var, 0) + delta
    if d[var] == 0:
        del d[var]
    return frozenset(d.items())


def _poly_add(a: Poly, b: Poly, sign: int = 1) -> Poly:
    out = dict(a)
    for m, c in b.items():
        out[m] = out.get(m, 0) + sign * c
        if out[m] == 0:
            del out[m]
    return out


def _poly_mul(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            d = dict(m1)
            for v, e in m2:
                d[v] = d.get(v, 0) + e
                if d[v] == 0:
                    del d[v]
            m = frozenset(d.items())
            out[m] = out.get(m, 0) + c1 * c2
            if out[m] == 0:
                del out[m]
    return out


def _is_single_monomial(poly: Poly) -> Tuple[bool, Union[None, Monom], Union[None, int]]:
    """Return (True, monom, coeff) if poly is exactly coeff * monom with one term."""
    if len(poly) != 1:
        return False, None, None
    ((m, c),) = poly.items()
    return True, m, c


def _poly_div_by_monomial(num: Poly, denom_m: Monom, denom_c: int) -> Union[Poly, None]:
    """
    Divide polynomial num by (denom_c * denom_m) if denom_c divides coefficients.
    We only allow division by a single monomial term.
    """
    if denom_c == 0:
        return None

    out: Poly = {}
    denom_exp = dict(denom_m)
    for m, c in num.items():
        if c % denom_c != 0:
            return None
        d = dict(m)
        # subtract exponents
        for v, e in denom_exp.items():
            d[v] = d.get(v, 0) - e
            if d[v] == 0:
                del d[v]
        nm = frozenset(d.items())
        nc = c // denom_c
        out[nm] = out.get(nm, 0) + nc
        if out[nm] == 0:
            del out[nm]
    return out


# ----------------------------
# Canonicalization
# ----------------------------


def _canonical_items_from_poly(poly: Poly) -> Tuple[Tuple[int, Tuple[Tuple[str, int], ...]], ...]:
    """Canonical hashable representation used for equality/dedup."""
    items: List[Tuple[int, Tuple[Tuple[str, int], ...]]] = []
    for m, c in poly.items():
        if c == 0:
            continue
        vars_exps = tuple(sorted(m))  # (var, exponent)
        items.append((c, vars_exps))
    items.sort()
    return tuple(items)


def _structural_key(expr: Union[str, Operation]) -> Tuple:
    """
    Fallback structural key (no algebra), still commutative for + and *.
    Useful if we hit unsupported constructs (e.g., division by a sum).
    """
    if isinstance(expr, Operation):
        op = expr.op
        lk = _structural_key(expr.left)
        rk = _structural_key(expr.right)
        if op in ("+", "*"):
            a, b = sorted((lk, rk))
            return (op, a, b)
        return (op, lk, rk)
    return ("var", str(expr))


def _expr_to_canonical_key(expr: Union[str, Operation]) -> Tuple:
    """
    Convert an Operation tree into a hashable canonical key that is invariant to:
    - Commutativity/associativity of * and +
    - Interaction of * and / via exponent bookkeeping
    """

    def to_poly(node: Union[str, Operation]) -> Union[Poly, None]:
        if isinstance(node, Operation):
            op = node.op
            left = to_poly(node.left)
            right = to_poly(node.right)
            if left is None or right is None:
                return None

            if op == "+":
                return _poly_add(left, right, sign=1)
            if op == "-":
                return _poly_add(left, right, sign=-1)
            if op == "*":
                return _poly_mul(left, right)
            if op == "/":
                ok, denom_m, denom_c = _is_single_monomial(right)
                if not ok:
                    return None
                # divide by denom_c * denom_m
                out = _poly_div_by_monomial(left, denom_m, denom_c)
                return out
            raise ValueError(f"Unknown operator {op!r}")
        else:
            # atomic variable name
            var = str(node)
            return {_monom_from_var(var, 1): 1}

    poly = to_poly(expr)
    if poly is None:
        # Non-monomial division or other unsupported form
        return ("struct", _structural_key(expr))

    return ("poly",) + (_canonical_items_from_poly(poly),)


def filter_canonical_expressions(exprs: list[Operation]) -> np.ndarray:
    """
    Return indices of the *first* occurrence of each canonical expression.
    keep='first' semantics.
    """
    seen: Dict[Hashable, int] = {}
    keep_indices: List[int] = []

    for i, expr in enumerate(exprs):
        key = _expr_to_canonical_key(expr)
        if key not in seen:
            seen[key] = i
            keep_indices.append(i)

    return np.asarray(keep_indices, dtype=int)
