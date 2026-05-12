from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TransientState(Protocol):
    n: int
    non_zero_ind: np.ndarray
    non_zero_val: np.ndarray

    def as_ndarray(self, coerce_dtype=None): ...

    def as_complex_ndarray(self): ...

    def scale(self, value): ...


@runtime_checkable
class SphericalHOBasisState(TransientState, Protocol):
    l: int
    m: int

    def as_ndarray_full_basis(self, coerce_dtype=None): ...

    def state_order(self): ...

    def get_indices(self): ...


@runtime_checkable
class AngularMomentumMatrices(Protocol):
    N: int
    basis_size: int

    def get_linear_operators(self, operator_buffer_size=None): ...

    def get_ladder_operators(self, operator_buffer_size=None): ...

    def get_casimir(self): ...


@runtime_checkable
class SphericalHOBasis(Protocol):
    Nmax: int
    basis_states: np.ndarray
    basis_states_views_n: list[np.ndarray]
    seed_states: dict

    def initialize_calculation(self): ...

    def calculate_states(self, method="Threads", max_workers=None): ...

    def job_make_block(self, n): ...

    def job_make_seed_state(self, n, l): ...

    def record_seed_state(self, seed_state): ...

    def complete_fixed_n_block_from_seed_states(self, seed_states): ...

    def as_complex_basis_states(self): ...

    def as_complex_basis_states_views_n(self): ...

    def s2_operator(self, state_in): ...

    def r22_operator(self, state_in): ...


__all__ = [
    "AngularMomentumMatrices",
    "SphericalHOBasis",
    "SphericalHOBasisState",
    "TransientState",
]
