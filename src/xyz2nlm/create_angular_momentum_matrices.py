import numpy as np
import mlx_create_c
from scipy.sparse.linalg import LinearOperator
from scipy.special import binom

class CreateAngularMomentumMatrices:
    def __init__(self, N):
        self.N = N
        if not self.N == 0:
            self._calculate_mapping_matrices()

    def get_linear_operators(self, operator_buffer_size=None):
        if self.N == 0:
            zero_array = np.zeros((1,1), dtype=complex)
            return zero_array, zero_array, zero_array

        Lx = self._package_in_linear_operator("Lx", ((1, 2, -1j), (2, 1, 1j)), operator_buffer_size)
        Ly = self._package_in_linear_operator("Ly", ((2, 0, -1j), (0, 2, 1j)), operator_buffer_size)
        Lz = self._package_in_linear_operator("Lz", ((0, 1, -1j), (1, 0, 1j)), operator_buffer_size)
        return Lx, Ly, Lz

    def get_ladder_operators(self, operator_buffer_size=None):
        if self.N == 0:
            zero_array = np.zeros((1,1), dtype=complex)
            return zero_array, zero_array

        Lp = self._package_in_linear_operator("Lp",
                                              ((1, 2, -1j), (2, 1, 1j),
                                               (2, 0, +1), (0, 2, -1)),
                                              operator_buffer_size)
        Lm = self._package_in_linear_operator("Lm",
                                              ((1, 2, -1j), (2, 1, 1j),
                                               (2, 0, -1), (0, 2, +1)),
                                              operator_buffer_size)
        return Lp, Lm

    def get_casimir(self):
        if self.N == 0:
            zero_array = np.zeros((1,1), dtype=complex)
            return zero_array

        Lx, Ly, Lz = self.get_linear_operators()
        function = lambda vec: Lx @ (Lx @ vec) + Ly @ (Ly @ vec) + Lz @ (Lz @ vec)
        L2 = LinearOperator((self.basis_size, self.basis_size),
                              matvec=function,
                              matmat=function)
        return L2

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

    def _create_operator_buffers(self, name, buffer_size):

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
                cleaner_function = self._create_operator_buffers(name, number_of_entries)
                return cleaner_function(psi)

            result_buffer[:, :number_of_entries].fill(0)
            return result_buffer[:, :number_of_entries].reshape(psi_shape)

        return clean_buffer_of_appropriate_size

    def _change_buffer_of_operator(self, name, new_buffer):
        buffer_name = f"_{name}_buffer"
        setattr(self, buffer_name, new_buffer)

    def _create_Hpsi(self, name, terms, operator_buffer_size=None):

        maps = []
        for term in terms:
            lhs_map_term, rhs_map_term, coefficients_term = \
                self._calculate_linear_operator_elements(term[0], term[1])
            coefficients_term = term[2] * coefficients_term.astype(complex)
            maps.append([lhs_map_term, rhs_map_term, coefficients_term])
            
        if operator_buffer_size is not None:
            clean_buffer = self._create_operator_buffers(name, operator_buffer_size)
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

    def _package_in_linear_operator(self, name, terms, operator_buffer_size=None):
        matvec_fun, matmat_fun = self._create_Hpsi(name, terms, operator_buffer_size)
        return LinearOperator((self.basis_size, self.basis_size),
                              matvec=matvec_fun,
                              matmat=matmat_fun)

if __name__ == "__main__":
    print('''This file provides the definition of the 
CreateAngularMomentumMatrices class and is not intended to be run as a script''')
