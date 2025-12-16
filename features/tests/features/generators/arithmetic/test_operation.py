# tests/test_canonical_key.py

from autogluon.features.generators.arithmetic.canonical_key import _expr_to_canonical_key  # adjust import path
from autogluon.features.generators.arithmetic.operation import Operation


def test_expr_to_canonical_key_commutative_mul():
    # A*B == B*A
    k1 = _expr_to_canonical_key(Operation("A", "B", "*"))
    k2 = _expr_to_canonical_key(Operation("B", "A", "*"))
    assert k1 == k2


def test_expr_to_canonical_key_commutative_add():
    # A+B == B+A
    k1 = _expr_to_canonical_key(Operation("A", "B", "+"))
    k2 = _expr_to_canonical_key(Operation("B", "A", "+"))
    assert k1 == k2


def test_expr_to_canonical_key_associative_mul():
    # (A*B)*C == A*(B*C)
    k1 = _expr_to_canonical_key(Operation(Operation("A", "B", "*"), "C", "*"))
    k2 = _expr_to_canonical_key(Operation("A", Operation("B", "C", "*"), "*"))
    assert k1 == k2


def test_expr_to_canonical_key_associative_add():
    # (A+B)+C == A+(B+C)
    k1 = _expr_to_canonical_key(Operation(Operation("A", "B", "+"), "C", "+"))
    k2 = _expr_to_canonical_key(Operation("A", Operation("B", "C", "+"), "+"))
    assert k1 == k2


def test_expr_to_canonical_key_mul_div_interaction():
    # (A/B)*C == (C*A)/B
    expr1 = Operation(Operation("A", "B", "/"), "C", "*")
    expr2 = Operation(Operation("C", "A", "*"), "B", "/")
    k1 = _expr_to_canonical_key(expr1)
    k2 = _expr_to_canonical_key(expr2)
    assert k1 == k2


def test_expr_to_canonical_key_subtraction_not_commutative():
    # A-B != B-A
    k1 = _expr_to_canonical_key(Operation("A", "B", "-"))
    k2 = _expr_to_canonical_key(Operation("B", "A", "-"))
    assert k1 != k2


def test_expr_to_canonical_key_fallback_non_monomial_division():
    # (A+B)/C cannot be reduced as polynomial-by-monomial division in the implementation
    # but should still produce a stable key via structural fallback.
    expr1 = Operation(Operation("A", "B", "+"), "C", "/")
    expr2 = Operation("C", Operation("A", "B", "+"), "/")  # different structure
    k1 = _expr_to_canonical_key(expr1)
    k2 = _expr_to_canonical_key(expr2)
    assert k1 != k2

    # And commutativity inside the numerator should still be respected by structural key
    expr3 = Operation(Operation("B", "A", "+"), "C", "/")
    k3 = _expr_to_canonical_key(expr3)
    assert k1 == k3
