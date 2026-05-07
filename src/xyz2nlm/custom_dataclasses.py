import numpy as np
import mlx_create_c
from dataclasses import dataclass

@dataclass(frozen=True)
class BaseStateDataClass:
    n : int
    non_zero_ind : np.ndarray
    non_zero_val : np.ndarray

    def as_ndarray(self, coerce_dtype=None):
        Nstates = ((self.n + 1) * (self.n + 2))//2

        if coerce_dtype is not None:
            out = np.zeros((Nstates,), dtype=coerce_dtype)
        else:
            out = np.zeros((Nstates,), dtype=self.non_zero_val.dtype)

        out[self.non_zero_ind] = self.non_zero_val
        return out

    def print_element(self):
        order = np.argsort(self.non_zero_val)
        Ns = mlx_create_c.create_bos_ns(self.n, 3)
        for element in order:
            print(f"nx={Ns[self.non_zero_ind[element],0]}, "
                  f"ny={Ns[self.non_zero_ind[element],1]}, "
                  f"nz={Ns[self.non_zero_ind[element],2]}: "
                  f"value={self.non_zero_val[element]}")

@dataclass(frozen=True)
class TransientState(BaseStateDataClass):

    @classmethod
    def from_ndarray(cls, n, array):
        if not isinstance(array, np.ndarray):
            raise ValueError("Data provided is not a np.ndarray")

        array = array.flatten()

        Nstates = ((n + 1) * (n + 2))//2
        Nelements = array.shape[0]
        if Nelements != Nstates:
            raise ValueError("The provided array does not have the appropriate number of "
                             f"elements for a state of rank {n}: expected{Nstates}, got "
                             f"{Nelements}.")

        return TransientState(n=n,
                              non_zero_ind=np.nonzero(array)[0],
                              non_zero_val=array[array!=0])

    def scale(self, value):
        return TransientState(n=self.n,
                              non_zero_ind=self.non_zero_ind,
                              non_zero_val=value*self.non_zero_val)

    def _set_sum(self, other):
        common_ind, pos_common_self, pos_common_other = \
            np.intersect1d(self.non_zero_ind, other.non_zero_ind,
                           assume_unique=True,
                           return_indices=True)

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

        non_zero_ind_out[n_unique_self:(n_unique_self+n_unique_other)] = \
            other.non_zero_ind[mask_unique_other]
        non_zero_val_out[n_unique_self:(n_unique_self+n_unique_other)] = \
            other.non_zero_val[mask_unique_other]

        non_zero_ind_out[(n_unique_self+n_unique_other):] = \
            common_ind
        non_zero_val_out[(n_unique_self+n_unique_other):] = \
            self.non_zero_val[pos_common_self] + other.non_zero_val[pos_common_other]

        return TransientState(n=self.n,
                              non_zero_ind=non_zero_ind_out,
                              non_zero_val=non_zero_val_out)

    def _array_sum(self, other):

        self_array = self.as_ndarray()
        other_array = other.as_ndarray()

        return self.from_ndarray(self.n, self_array + other_array)

    def __add__(self, other):
        """We prefer set sum because it scales better for large n"""
        if other == 0:
            return self

        if not isinstance(other, TransientState):
            raise ValueError("A TransientState can be summed only with another TransientState")

        if self.n != other.n:
            raise ValueError("A TransientState can be summed only with another TransientState"
                             " with the same value of n\nn_first {self.n} n_second {other.n}")

        return self._set_sum(other)

@dataclass(frozen=True)
class SphericalHOBasisState(BaseStateDataClass):
    l : int
    m : int

    def as_ndarray_full_basis(self, coerce_dtype=None):
        compressed = self.as_ndarray(dtype)
        previous_states = ((self.n) * (self.n + 1) * (self.n + 2))//6

        if coerce_dtype is not None:
            dtype=coerce_dtype
        else:
            dtype=self.non_zero_val.dtype

        return np.r_[np.zeros((previous_states,), dtype=dtype), compressed]

    def state_order(self):
        return self.n, (self.l * (self.l + 1))//2 + (self.l + self.m)//2

    def get_indices(self):
        return self.n, self.l, self.m

    # it could also get a from_transient_state method
