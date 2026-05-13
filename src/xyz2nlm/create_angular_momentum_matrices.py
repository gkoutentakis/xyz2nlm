from abc import ABC, abstractmethod

import numpy as np
import mlx_create_c
from scipy.sparse.linalg import LinearOperator

from .qi_interface import exact_zero_array, is_zero, qi, qi_i, qi_sqrt_int, qi_zero, to_complex


class ExactAngularMomentumOperator:
    def __init__(self, basis_size, maps, name, operator_buffer_size=None, owner=None):
        self.basis_size = basis_size
        self.shape = (basis_size, basis_size)
        self.maps = maps
        self.name = name
        self.owner = owner
        self.buffer = None
        self._adjacency_by_rhs = None
        if operator_buffer_size is not None:
            self.change_buffer(exact_zero_array((basis_size, operator_buffer_size)))

    def change_buffer(self, new_buffer):
        self.buffer = new_buffer
        if self.owner is not None:
            setattr(self.owner, f"_{self.name}_buffer", new_buffer)

    def adjacency_by_rhs(self):
        if self._adjacency_by_rhs is not None:
            return self._adjacency_by_rhs

        adjacency = [[] for _ in range(self.basis_size)]
        for lhs_map, rhs_map, coefficients, _numeric_coefficients in self.maps:
            for lhs, rhs, coefficient in zip(lhs_map, rhs_map, coefficients):
                if not is_zero(coefficient):
                    adjacency[int(rhs)].append((int(lhs), coefficient))

        self._adjacency_by_rhs = tuple(tuple(entries) for entries in adjacency)
        return self._adjacency_by_rhs

    def apply_sparse_columns(self, sparse_columns):
        adjacency = self.adjacency_by_rhs()
        output_columns = []

        for column in sparse_columns:
            output = {}
            for rhs, value in column.items():
                for lhs, coefficient in adjacency[rhs]:
                    contribution = coefficient * value
                    if is_zero(contribution):
                        continue

                    updated_value = output.get(lhs, qi_zero()) + contribution
                    if is_zero(updated_value):
                        output.pop(lhs, None)
                    else:
                        output[lhs] = updated_value

            output_columns.append(output)

        return output_columns

    def _clean_exact_output(self, psi):
        if self.buffer is None:
            return exact_zero_array(psi.shape)

        psi_shape = psi.shape
        if psi_shape[0] != self.basis_size:
            raise ValueError(
                f"The provided state cannot be multiplied with {self.name}: "
                f"got state.shape={psi_shape} and operator shape={self.shape}"
            )

        if len(psi_shape) == 1:
            view = self.buffer[:, 0]
            view[:] = qi_zero()
            return view

        number_of_entries = int(np.prod(psi_shape[1:]))
        if number_of_entries > self.buffer.shape[1]:
            self.change_buffer(exact_zero_array((self.basis_size, number_of_entries)))

        out = self.buffer[:, :number_of_entries].reshape(psi_shape)
        out[:] = qi_zero()
        return out

    def _apply_exact(self, psi):
        psi = np.asarray(psi, dtype=object)
        psi_shape = psi.shape
        if psi_shape[0] != self.basis_size:
            raise ValueError(
                f"The provided state cannot be multiplied with {self.name}: "
                f"got state.shape={psi_shape} and operator shape={self.shape}"
            )

        out = self._clean_exact_output(psi)
        psi_flat = psi.reshape((self.basis_size, -1))
        out_flat = out.reshape((self.basis_size, -1))
        zero = qi_zero()

        for lhs_map, rhs_map, coefficients, _numeric_coefficients in self.maps:
            for lhs, rhs, coefficient in zip(lhs_map, rhs_map, coefficients):
                for column in range(psi_flat.shape[1]):
                    value = psi_flat[rhs, column]
                    if value == zero:
                        continue
                    out_flat[lhs, column] = out_flat[lhs, column] + coefficient * value

        return out

    def _apply_numeric(self, psi):
        psi = np.asarray(psi)
        psi_shape = psi.shape
        if psi_shape[0] != self.basis_size:
            raise ValueError(
                f"The provided state cannot be multiplied with {self.name}: "
                f"got state.shape={psi_shape} and operator shape={self.shape}"
            )

        out = np.zeros_like(psi, dtype=complex)
        shape = (-1,) + (1,) * (psi.ndim - 1)
        for lhs_map, rhs_map, _coefficients, numeric_coefficients in self.maps:
            out[lhs_map, ...] += numeric_coefficients.reshape(shape) * psi[rhs_map, ...]
        return out

    def matvec(self, psi):
        psi = np.asarray(psi)
        if psi.dtype == object:
            return self._apply_exact(psi)
        return self._apply_numeric(psi)

    def matmat(self, mat):
        mat = np.asarray(mat)
        if mat.dtype == object:
            return self._apply_exact(mat)
        return self._apply_numeric(mat)

    def __matmul__(self, other):
        if hasattr(other, "matvec") and hasattr(other, "matmat"):
            return ExactComposedOperator(self.basis_size, [self, other])

        other = np.asarray(other)
        if other.ndim == 1:
            return self.matvec(other)
        return self.matmat(other)


class ExactComposedOperator:
    def __init__(self, basis_size, operators):
        self.basis_size = basis_size
        self.shape = (basis_size, basis_size)
        self.operators = operators

    def matvec(self, psi):
        out = psi
        for operator in reversed(self.operators):
            out = operator @ out
        return out

    def matmat(self, mat):
        return self.matvec(mat)

    def __matmul__(self, other):
        if hasattr(other, "matvec") and hasattr(other, "matmat"):
            return ExactComposedOperator(self.basis_size, self.operators + [other])
        return self.matvec(other)


class ExactCasimirOperator:
    def __init__(self, basis_size, lx, ly, lz):
        self.basis_size = basis_size
        self.shape = (basis_size, basis_size)
        self.lx = lx
        self.ly = ly
        self.lz = lz

    def matvec(self, psi):
        result = self.lx @ (self.lx @ psi)
        result = result + self.ly @ (self.ly @ psi)
        result = result + self.lz @ (self.lz @ psi)
        return result

    def matmat(self, mat):
        return self.matvec(mat)

    def __matmul__(self, other):
        if hasattr(other, "matvec") and hasattr(other, "matmat"):
            return ExactComposedOperator(self.basis_size, [self, other])
        return self.matvec(other)


class _AngularMomentumMatricesBase(ABC):

    def __init__(self, N):
        self.N = int(N)
        self.basis_size = ((self.N + 1) * (self.N + 2)) // 2
        if self.N != 0:
            self._calculate_mapping_matrices()
        else:
            self.Ns1 = np.empty((0, 3), dtype=np.int64)
            self.mapmat = np.empty((0, 3), dtype=np.int64)

    @abstractmethod
    def get_linear_operators(self, operator_buffer_size=None):
        pass

    @abstractmethod
    def get_ladder_operators(self, operator_buffer_size=None):
        pass

    @abstractmethod
    def get_casimir(self):
        pass

    def _calculate_mapping_matrices(self):
        self.Ns1 = mlx_create_c.create_bos_ns(self.N-1, 3)
        self.mapmat = mlx_create_c.create_mapmat_bos(self.Ns1)
        self.basis_size = self.mapmat[-1,-1] + 1

    def _calculate_linear_operator_elements(self, i, j):
        mapmat = self.mapmat.view()
        Ns1 = self.Ns1.view()

        lhs_map = mapmat[:, i]
        rhs_map = mapmat[:, j]
        coefficients = np.sqrt((Ns1[:, i] + 1.) * (Ns1[:, j] + 1.))

        return lhs_map, rhs_map, coefficients

    def _calculate_exact_operator_elements(self, i, j):
        if self.N == 0:
            empty = np.asarray([], dtype=np.int64)
            return empty, empty, np.asarray([], dtype=object)

        lhs_map = self.mapmat[:, i]
        rhs_map = self.mapmat[:, j]
        coefficients = np.asarray(
            [qi_sqrt_int((row[i] + 1) * (row[j] + 1)) for row in self.Ns1],
            dtype=object,
        )
        return lhs_map, rhs_map, coefficients

    def _create_operator_buffers_np(self, name, buffer_size):

        if buffer_size < 1:
            raise ValueError(f"size of buffers should be positive integer, not {buffer_size}")

        buffer_name = f"_{name}_buffer"
        result_buffer = np.empty((self.basis_size, buffer_size), dtype=complex)
        setattr(self, buffer_name, result_buffer)

        def clean_buffer_of_appropriate_size(psi):
            result_buffer = getattr(self, buffer_name)
            buffer_size = result_buffer.shape[1]

            psi_shape = psi.shape

            if psi_shape[0] != self.basis_size:
                raise ValueError(
                    f"The provided state cannot be multiplied with the operator (name):\n"
                    f"Incompatible shapes state.shape={psi_shape} and "
                    f"{name}.shape={self.basis_size}x{self.basis_size}")

            if len(psi_shape) == 1:
                view = result_buffer[:, 0]
                view.fill(0)
                return view

            number_of_entries = int(np.prod(psi_shape[1:]))
            if number_of_entries > buffer_size:
                cleaner_function = self._create_operator_buffers_np(name, number_of_entries)
                return cleaner_function(psi)

            result_buffer[:, :number_of_entries].fill(0)
            return result_buffer[:, :number_of_entries].reshape(psi_shape)

        return clean_buffer_of_appropriate_size

    def _change_buffer_of_operator(self, name, new_buffer):
        buffer_name = f"_{name}_buffer"
        setattr(self, buffer_name, new_buffer)
        operator = getattr(self, f"_{name}_operator", None)
        if operator is not None and hasattr(operator, "change_buffer"):
            operator.change_buffer(new_buffer)

    def _create_Hpsi_np(self, name, terms, operator_buffer_size=None):

        maps = []
        for term in terms:
            lhs_map_term, rhs_map_term, coefficients_term = \
                self._calculate_linear_operator_elements(term[0], term[1])
            coefficients_term = term[2] * coefficients_term.astype(complex)
            maps.append([lhs_map_term, rhs_map_term, coefficients_term])
            
        if operator_buffer_size is not None:
            clean_buffer = self._create_operator_buffers_np(name, operator_buffer_size)
        else:
            clean_buffer = lambda psi: np.zeros_like(psi, dtype=complex)

        def _hpsi(psi):
            psi_shape = psi.shape
            psi = psi.flatten()

            hpsi = clean_buffer(psi)
            for mapt in maps:
                hpsi[mapt[0]] += mapt[2] * psi[mapt[1]]
            return hpsi.reshape(psi_shape)
        setattr(self, f"_{name}_hpsi", _hpsi)

        def _hmat(mat):
            hmat = clean_buffer(mat)
            shape = (-1,) + (1,) * (mat.ndim - 1)
            
            for mapt in maps:
                hmat[mapt[0], ...] += mapt[2].reshape(shape) * mat[mapt[1], ...]

            return hmat
        setattr(self, f"_{name}_hmat", _hmat)

        return _hpsi, _hmat

    def _package_in_linear_operator_np(self, name, terms, operator_buffer_size=None):
        matvec_fun, matmat_fun = self._create_Hpsi_np(name, terms, operator_buffer_size)
        return LinearOperator((self.basis_size, self.basis_size),
                              matvec=matvec_fun,
                              matmat=matmat_fun)

    def _package_in_exact_operator(self, name, terms, operator_buffer_size=None):
        maps = []
        for term in terms:
            lhs_map, rhs_map, coefficients = self._calculate_exact_operator_elements(term[0], term[1])
            coefficients = np.asarray([term[2] * coefficient for coefficient in coefficients], dtype=object)
            numeric_coefficients = np.asarray([to_complex(coefficient) for coefficient in coefficients])
            maps.append((lhs_map, rhs_map, coefficients, numeric_coefficients))

        operator = ExactAngularMomentumOperator(
            self.basis_size,
            maps,
            name,
            operator_buffer_size=operator_buffer_size,
            owner=self,
        )
        setattr(self, f"_{name}_operator", operator)
        return operator


class AngularMomentumMatricesQI(_AngularMomentumMatricesBase):

    def get_linear_operators(self, operator_buffer_size=None):
        Lx = self._package_in_exact_operator("Lx", ((1, 2, -qi_i()), (2, 1, qi_i())), operator_buffer_size)
        Ly = self._package_in_exact_operator("Ly", ((2, 0, -qi_i()), (0, 2, qi_i())), operator_buffer_size)
        Lz = self._package_in_exact_operator("Lz", ((0, 1, -qi_i()), (1, 0, qi_i())), operator_buffer_size)
        return Lx, Ly, Lz

    def get_ladder_operators(self, operator_buffer_size=None):
        Lp = self._package_in_exact_operator(
            "Lp",
            ((1, 2, -qi_i()), (2, 1, qi_i()), (2, 0, qi(1)), (0, 2, qi(-1))),
            operator_buffer_size,
        )
        Lm = self._package_in_exact_operator(
            "Lm",
            ((1, 2, -qi_i()), (2, 1, qi_i()), (2, 0, qi(-1)), (0, 2, qi(1))),
            operator_buffer_size,
        )
        return Lp, Lm

    def get_casimir(self):
        Lx, Ly, Lz = self.get_linear_operators()
        return ExactCasimirOperator(self.basis_size, Lx, Ly, Lz)


class AngularMomentumMatricesNP(_AngularMomentumMatricesBase):

    def get_linear_operators(self, operator_buffer_size=None):
        if self.N == 0:
            zero_array = np.zeros((1,1), dtype=complex)
            return zero_array, zero_array, zero_array

        Lx = self._package_in_linear_operator_np("Lx", ((1, 2, -1j), (2, 1, 1j)), operator_buffer_size)
        Ly = self._package_in_linear_operator_np("Ly", ((2, 0, -1j), (0, 2, 1j)), operator_buffer_size)
        Lz = self._package_in_linear_operator_np("Lz", ((0, 1, -1j), (1, 0, 1j)), operator_buffer_size)
        return Lx, Ly, Lz

    def get_ladder_operators(self, operator_buffer_size=None):
        if self.N == 0:
            zero_array = np.zeros((1,1), dtype=complex)
            return zero_array, zero_array

        Lp = self._package_in_linear_operator_np(
            "Lp",
            ((1, 2, -1j), (2, 1, 1j), (2, 0, +1), (0, 2, -1)),
            operator_buffer_size,
        )
        Lm = self._package_in_linear_operator_np(
            "Lm",
            ((1, 2, -1j), (2, 1, 1j), (2, 0, -1), (0, 2, +1)),
            operator_buffer_size,
        )
        return Lp, Lm

    def get_casimir(self):
        if self.N == 0:
            return np.zeros((1,1), dtype=complex)

        Lx, Ly, Lz = self.get_linear_operators()
        function = lambda vec: Lx @ (Lx @ vec) + Ly @ (Ly @ vec) + Lz @ (Lz @ vec)
        return LinearOperator((self.basis_size, self.basis_size),
                              matvec=function,
                              matmat=function)


class _BlockLmApplicationBase(ABC):
    angular_momentum_class = None

    def __init__(self, n, block_in, ls, ms, seed_states=None):
        self.am = self.angular_momentum_class(n)
        _, self.Lm = self.am.get_ladder_operators(operator_buffer_size=self.am.basis_size)

        self.work_array_a = block_in
        self.work_array_b = self.am._Lm_buffer

        self.states_current = self.work_array_a
        self.states_current_using_buffer_a = True

        self.block_ls = ls.copy()
        self.block_ms = ms.copy()

    def decrease_m_of_block(self):
        self.states_current = self.Lm @ self.states_current
        self.update_working_buffer_data()
        self.normalize_states_after_lm_application()

        self.block_ms -= 1

        return self.states_current, self.block_ls, self.block_ms

    def update_working_buffer_data(self):
        if self.states_current_using_buffer_a:
            self.am._change_buffer_of_operator("Lm", self.work_array_a)
            self.states_current_using_buffer_a = False
        else:
            self.am._change_buffer_of_operator("Lm", self.work_array_b)
            self.states_current_using_buffer_a = True

    @abstractmethod
    def normalize_states_after_lm_application(self):
        pass

    def drop_first_state_from_block(self):
        self.states_current = self.states_current[:, 1:]
        self.block_ls = self.block_ls[1:]
        self.block_ms = self.block_ms[1:]


class BlockLmApplicationQI(_BlockLmApplicationBase):
    angular_momentum_class = AngularMomentumMatricesQI

    def __init__(self, n, block_in, ls, ms, seed_states=None):
        self.am = self.angular_momentum_class(n)
        _, self.Lm = self.am.get_ladder_operators()

        self.work_array_a = block_in
        self.work_array_b = None
        self.states_current = self.work_array_a
        self.states_current_using_buffer_a = True

        self.block_ls = ls.copy()
        self.block_ms = ms.copy()
        self.sparse_columns = self._initial_sparse_columns(block_in, seed_states)

    def _initial_sparse_columns(self, block_in, seed_states):
        if seed_states is not None:
            return [
                {
                    int(index): value
                    for index, value in zip(state.non_zero_ind, state.non_zero_val)
                    if not is_zero(value)
                }
                for state in seed_states
            ]

        return self._sparse_columns_from_dense(block_in)

    def _sparse_columns_from_dense(self, block_in):
        block_in = np.asarray(block_in, dtype=object)
        sparse_columns = []
        for column in range(block_in.shape[1]):
            sparse_column = {}
            for row, value in enumerate(block_in[:, column]):
                if not is_zero(value):
                    sparse_column[row] = value
            sparse_columns.append(sparse_column)
        return sparse_columns

    def decrease_m_of_block(self):
        self.sparse_columns = self.Lm.apply_sparse_columns(self.sparse_columns)
        self.normalize_states_after_lm_application()
        self.block_ms -= 1
        self.states_current = self._materialize_sparse_columns()
        return self.states_current, self.block_ls, self.block_ms

    def normalize_states_after_lm_application(self):
        for column_index, (ell, emm) in enumerate(zip(self.block_ls, self.block_ms)):
            normalization = qi_sqrt_int((ell + emm) * (ell - emm + 1))
            normalized_column = {}
            for index, value in self.sparse_columns[column_index].items():
                normalized_value = value / normalization
                if not is_zero(normalized_value):
                    normalized_column[index] = normalized_value

            self.sparse_columns[column_index] = normalized_column

    def _materialize_sparse_columns(self):
        out = exact_zero_array((self.am.basis_size, len(self.sparse_columns)))
        for column_index, sparse_column in enumerate(self.sparse_columns):
            for row, value in sparse_column.items():
                out[row, column_index] = value
        return out

    def drop_first_state_from_block(self):
        self.sparse_columns = self.sparse_columns[1:]
        self.states_current = self._materialize_sparse_columns()
        self.block_ls = self.block_ls[1:]
        self.block_ms = self.block_ms[1:]


class BlockLmApplicationNP(_BlockLmApplicationBase):
    angular_momentum_class = AngularMomentumMatricesNP

    def normalize_states_after_lm_application(self):
        self.states_current /= np.sqrt(
            (self.block_ls + self.block_ms) * (self.block_ls - self.block_ms + 1)
        )[None, :]


def create_angular_momentum_matrices(N, exact=True, compatible_basis=None):
    if compatible_basis:
        implementation = AngularMomentumMatricesQI \
            if compatible_basis.block_application_class is BlockLmApplicationQI \
               else AngularMomentumMatricesNP
    else:
        implementation = AngularMomentumMatricesQI if exact else AngularMomentumMatricesNP

    return implementation(N)


if __name__ == "__main__":
    print('''This file provides the definition of the 
AngularMomentumMatrices class and is not intended to be run as a script''')
