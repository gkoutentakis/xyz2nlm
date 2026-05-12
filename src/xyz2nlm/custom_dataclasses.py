from dataclasses import dataclass

import mlx_create_c
import numpy as np

from .qi_algebra import QINumber
from .qi_interface import exact_zero_array, is_zero, qi, qi_zero, to_complex_array


def _sector_size(n):
    return ((n + 1) * (n + 2)) // 2


def _values_need_qi_backend(values):
    values = np.asarray(values)
    if values.dtype != object:
        return False
    return any(isinstance(value, QINumber) for value in values.flat)


@dataclass(frozen=True)
class BaseStateDataClass:
    n: int
    non_zero_ind: np.ndarray
    non_zero_val: np.ndarray

    def as_ndarray(self, coerce_dtype=None):
        nstates = _sector_size(self.n)
        dtype = coerce_dtype if coerce_dtype is not None else self.non_zero_val.dtype
        out = np.zeros((nstates,), dtype=dtype)
        out[self.non_zero_ind] = self.non_zero_val
        return out

    def as_complex_ndarray(self):
        return self.as_ndarray(coerce_dtype=complex)

    def print_element(self):
        order = np.argsort(self.non_zero_ind)
        ns = mlx_create_c.create_bos_ns(self.n, 3)
        for element in order:
            print(
                f"nx={ns[self.non_zero_ind[element], 0]}, "
                f"ny={ns[self.non_zero_ind[element], 1]}, "
                f"nz={ns[self.non_zero_ind[element], 2]}: "
                f"value={self.non_zero_val[element]}"
            )


@dataclass(frozen=True)
class TransientStateNP(BaseStateDataClass):
    @classmethod
    def from_ndarray(cls, n, array):
        if not isinstance(array, np.ndarray):
            raise ValueError("Data provided is not a np.ndarray")

        array = array.flatten()
        nstates = _sector_size(n)
        nelements = array.shape[0]
        if nelements != nstates:
            raise ValueError(
                "The provided array does not have the appropriate number of "
                f"elements for a state of rank {n}: expected {nstates}, got "
                f"{nelements}."
            )

        non_zero = np.nonzero(array)[0]
        return cls(n=n, non_zero_ind=non_zero, non_zero_val=array[non_zero])

    def scale(self, value):
        return type(self)(
            n=self.n,
            non_zero_ind=self.non_zero_ind,
            non_zero_val=value * self.non_zero_val,
        )

    def _set_sum(self, other):
        common_ind, pos_common_self, pos_common_other = np.intersect1d(
            self.non_zero_ind,
            other.non_zero_ind,
            assume_unique=True,
            return_indices=True,
        )

        mask_unique_self = np.ones(self.non_zero_ind.shape, dtype=bool)
        mask_unique_self[pos_common_self] = False

        mask_unique_other = np.ones(other.non_zero_ind.shape, dtype=bool)
        mask_unique_other[pos_common_other] = False

        n_common = common_ind.shape[0]
        n_unique_self = self.non_zero_ind.shape[0] - n_common
        n_unique_other = other.non_zero_ind.shape[0] - n_common

        nnz = n_unique_self + n_unique_other + n_common
        output_val_dtype = np.result_type(self.non_zero_val, other.non_zero_val)
        non_zero_ind_out = np.zeros((nnz,), dtype=self.non_zero_ind.dtype)
        non_zero_val_out = np.zeros((nnz,), dtype=output_val_dtype)

        non_zero_ind_out[:n_unique_self] = self.non_zero_ind[mask_unique_self]
        non_zero_val_out[:n_unique_self] = self.non_zero_val[mask_unique_self]

        non_zero_ind_out[n_unique_self : (n_unique_self + n_unique_other)] = (
            other.non_zero_ind[mask_unique_other]
        )
        non_zero_val_out[n_unique_self : (n_unique_self + n_unique_other)] = (
            other.non_zero_val[mask_unique_other]
        )

        non_zero_ind_out[(n_unique_self + n_unique_other) :] = common_ind
        non_zero_val_out[(n_unique_self + n_unique_other) :] = (
            self.non_zero_val[pos_common_self] + other.non_zero_val[pos_common_other]
        )

        non_zero = non_zero_val_out != 0
        return type(self)(
            n=self.n,
            non_zero_ind=non_zero_ind_out[non_zero],
            non_zero_val=non_zero_val_out[non_zero],
        )

    def _array_sum(self, other):
        self_array = self.as_ndarray()
        other_array = other.as_ndarray()
        return type(self).from_ndarray(self.n, self_array + other_array)

    def __add__(self, other):
        if other == 0:
            return self

        if not isinstance(other, type(self)):
            raise ValueError(f"{type(self).__name__} can only be summed with the same type")

        if self.n != other.n:
            raise ValueError(
                f"{type(self).__name__} can only be summed with the same n: "
                f"got {self.n} and {other.n}"
            )

        return self._set_sum(other)


@dataclass(frozen=True)
class SphericalHOBasisStateNP(TransientStateNP):
    l: int
    m: int

    def as_ndarray_full_basis(self, coerce_dtype=None):
        compressed = self.as_ndarray(coerce_dtype=coerce_dtype)
        previous_states = (self.n * (self.n + 1) * (self.n + 2)) // 6
        dtype = coerce_dtype if coerce_dtype is not None else self.non_zero_val.dtype
        return np.r_[np.zeros((previous_states,), dtype=dtype), compressed]

    def state_order(self):
        return self.n, (self.l * (self.l + 1)) // 2 + (self.l + self.m) // 2

    def get_indices(self):
        return self.n, self.l, self.m


@dataclass(frozen=True)
class TransientStateQI(BaseStateDataClass):
    def __post_init__(self):
        non_zero_ind = np.asarray(self.non_zero_ind, dtype=np.int64)
        values = np.asarray([qi(value) for value in self.non_zero_val], dtype=object)

        if non_zero_ind.shape[0] != values.shape[0]:
            raise ValueError("non_zero_ind and non_zero_val must have equal lengths")

        if non_zero_ind.shape[0] == 0:
            object.__setattr__(self, "non_zero_ind", non_zero_ind)
            object.__setattr__(self, "non_zero_val", values)
            return

        accumulator = {}
        for index, value in zip(non_zero_ind, values):
            index = int(index)
            accumulator[index] = accumulator.get(index, qi_zero()) + value

        indices = []
        packed_values = []
        for index in sorted(accumulator):
            value = accumulator[index]
            if not is_zero(value):
                indices.append(index)
                packed_values.append(value)

        object.__setattr__(self, "non_zero_ind", np.asarray(indices, dtype=np.int64))
        object.__setattr__(self, "non_zero_val", np.asarray(packed_values, dtype=object))

    @classmethod
    def from_ndarray(cls, n, array):
        if not isinstance(array, np.ndarray):
            raise ValueError("Data provided is not a np.ndarray")

        array = array.flatten()
        nstates = _sector_size(n)
        nelements = array.shape[0]
        if nelements != nstates:
            raise ValueError(
                "The provided array does not have the appropriate number of "
                f"elements for a state of rank {n}: expected {nstates}, got "
                f"{nelements}."
            )

        indices = []
        values = []
        for index, value in enumerate(array):
            value = qi(value)
            if not is_zero(value):
                indices.append(index)
                values.append(value)

        return cls(n=n, non_zero_ind=np.asarray(indices), non_zero_val=np.asarray(values, dtype=object))

    def as_ndarray(self, coerce_dtype=None):
        if coerce_dtype is not None:
            return self.as_complex_ndarray().astype(coerce_dtype, copy=False)

        out = exact_zero_array((_sector_size(self.n),))
        out[self.non_zero_ind] = self.non_zero_val
        return out

    def as_complex_ndarray(self):
        out = np.zeros((_sector_size(self.n),), dtype=complex)
        out[self.non_zero_ind] = to_complex_array(self.non_zero_val)
        return out

    def scale(self, value):
        value = qi(value)
        return type(self)(
            n=self.n,
            non_zero_ind=self.non_zero_ind,
            non_zero_val=np.asarray([value * coeff for coeff in self.non_zero_val], dtype=object),
        )

    def _set_sum(self, other):
        accumulator = {}
        for index, value in zip(self.non_zero_ind, self.non_zero_val):
            accumulator[int(index)] = value
        for index, value in zip(other.non_zero_ind, other.non_zero_val):
            index = int(index)
            accumulator[index] = accumulator.get(index, qi_zero()) + value

        indices = []
        values = []
        for index in sorted(accumulator):
            value = accumulator[index]
            if not is_zero(value):
                indices.append(index)
                values.append(value)

        return type(self)(
            n=self.n,
            non_zero_ind=np.asarray(indices, dtype=np.int64),
            non_zero_val=np.asarray(values, dtype=object),
        )

    def _array_sum(self, other):
        return type(self).from_ndarray(self.n, self.as_ndarray() + other.as_ndarray())

    def __add__(self, other):
        if other == 0:
            return self

        if not isinstance(other, type(self)):
            raise ValueError(f"{type(self).__name__} can only be summed with the same type")

        if self.n != other.n:
            raise ValueError(
                f"{type(self).__name__} can only be summed with the same n: "
                f"got {self.n} and {other.n}"
            )

        return self._set_sum(other)


@dataclass(frozen=True)
class SphericalHOBasisStateQI(TransientStateQI):
    l: int
    m: int

    def as_ndarray_full_basis(self, coerce_dtype=None):
        compressed = self.as_ndarray(coerce_dtype=coerce_dtype)
        previous_states = (self.n * (self.n + 1) * (self.n + 2)) // 6
        if coerce_dtype is not None:
            dtype = coerce_dtype
            zeros = np.zeros((previous_states,), dtype=dtype)
        else:
            zeros = exact_zero_array((previous_states,))
        return np.r_[zeros, compressed]

    def as_complex_ndarray_full_basis(self):
        return self.as_ndarray_full_basis(coerce_dtype=complex)

    def state_order(self):
        return self.n, (self.l * (self.l + 1)) // 2 + (self.l + self.m) // 2

    def get_indices(self):
        return self.n, self.l, self.m


class TransientStateFactory:
    def __new__(cls, n, non_zero_ind, non_zero_val):
        if cls is not TransientStateFactory:
            return super().__new__(cls)

        implementation = TransientStateQI if _values_need_qi_backend(non_zero_val) else TransientStateNP
        return implementation(n=n, non_zero_ind=non_zero_ind, non_zero_val=non_zero_val)

    @classmethod
    def from_ndarray(cls, n, array):
        implementation = TransientStateQI if np.asarray(array).dtype == object else TransientStateNP
        return implementation.from_ndarray(n, array)


class SphericalHOBasisStateFactory:
    def __new__(cls, n, l, m, non_zero_ind, non_zero_val):
        if cls is not SphericalHOBasisStateFactory:
            return super().__new__(cls)

        implementation = SphericalHOBasisStateQI if _values_need_qi_backend(non_zero_val) else SphericalHOBasisStateNP
        return implementation(
            n=n,
            l=l,
            m=m,
            non_zero_ind=non_zero_ind,
            non_zero_val=non_zero_val,
        )
