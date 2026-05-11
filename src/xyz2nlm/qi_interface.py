import math

import numpy as np

from .qi_algebra import (
    QINumber,
    conjugate as qi_conjugate,
    is_zero as qi_is_zero,
    square_root_fraction,
    square_root_integer,
    to_complex as qi_to_complex,
)


def qi(value=0):
    if isinstance(value, QINumber):
        return value
    if isinstance(value, np.integer):
        return QINumber(int(value))
    if isinstance(value, np.floating):
        return QINumber(float(value))
    if isinstance(value, np.complexfloating):
        return QINumber(complex(value))
    return QINumber(value)


def qi_zero():
    return QINumber(0)


def qi_one():
    return QINumber(1)


def qi_i():
    return QINumber(1j)


def qi_minus_i():
    return -qi_i()


def qi_sqrt_int(value):
    return square_root_integer(int(value))


def qi_sqrt_fraction(numerator, denominator):
    return square_root_fraction(int(numerator), int(denominator))


def qi_phase_i(power):
    phase = int(power) % 4
    if phase == 0:
        return qi_one()
    if phase == 1:
        return qi_i()
    if phase == 2:
        return -qi_one()
    return qi_minus_i()


def qi_binom(n, k):
    return QINumber(math.comb(int(n), int(k)))


def is_zero(value):
    return qi_is_zero(qi(value))


def conjugate(value):
    return qi_conjugate(qi(value))


def to_complex(value):
    return qi_to_complex(qi(value))


def exact_zero_array(shape):
    out = np.empty(shape, dtype=object)
    out.fill(qi_zero())
    return out


def to_complex_array(array):
    array = np.asarray(array)
    out = np.zeros(array.shape, dtype=complex)
    for index in np.ndindex(array.shape):
        out[index] = to_complex(array[index])
    return out


def exact_inner_product(left, right):
    left_array = left.as_ndarray() if hasattr(left, "as_ndarray") else np.asarray(left)
    right_array = right.as_ndarray() if hasattr(right, "as_ndarray") else np.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"inner product requires equal shapes, got {left_array.shape} and {right_array.shape}"
        )

    total = qi_zero()
    for left_value, right_value in zip(left_array.flat, right_array.flat):
        if is_zero(left_value) or is_zero(right_value):
            continue
        total += conjugate(left_value) * qi(right_value)
    return total


def exact_norm(state):
    return exact_inner_product(state, state)


def exact_values_equal(left, right):
    return qi(left) == qi(right)
