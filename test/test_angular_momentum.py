import numpy as np
import numpy.testing as npt
import pytest

import xyz2nlm

@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("size", [4, 20])
def test_commutators(size, buffered):
    am = xyz2nlm.CreateAngularMomentumMatrices(size)

    if buffered:
        operator_buffer_size = am.basis_size
    else:
        operator_buffer_size = None

    Lx, Ly, Lz = am.get_linear_operators(operator_buffer_size)
    Lp, Lm = am.get_ladder_operators(operator_buffer_size)

    eye = np.eye(am.basis_size, dtype=complex)
    Lx_mat = Lx @ eye
    Ly_mat = Ly @ eye
    Lz_mat = Lz @ eye

    Lp_mat = Lp @ eye
    Lm_mat = Lm @ eye

    L2 = am.get_casimir()
    L2_mat = L2 @ eye

    # [Lx, Ly] = i \hbar Lz
    actual = Lx_mat @ Ly_mat - Ly_mat @ Lx_mat
    expected = 1j * Lz_mat
    npt.assert_allclose(actual, expected, atol=1e-12)

    # [Ly, Lz] = i \hbar Lx
    actual = Ly_mat @ Lz_mat - Lz_mat @ Ly_mat
    expected = 1j * Lx_mat
    npt.assert_allclose(actual, expected, atol=1e-12)

    # [Lz, Lx] = i \hbar Ly
    actual = Lz_mat @ Lx_mat - Lx_mat @ Lz_mat
    expected = 1j * Ly_mat
    npt.assert_allclose(actual, expected, atol=1e-12)

    # [Lz, L+] = + L+
    actual = Lz_mat @ Lp_mat - Lp_mat @ Lz_mat
    expected = Lp_mat
    npt.assert_allclose(actual, expected, atol=1e-12)

    # [Lz, L-] = - L-
    actual = Lz_mat @ Lm_mat - Lm_mat @ Lz_mat
    expected = -Lm_mat
    npt.assert_allclose(actual, expected, atol=1e-12)

    # [L+, L-] = 2 Lz
    actual = Lp_mat @ Lm_mat - Lm_mat @ Lp_mat
    expected = 2 * Lz_mat
    npt.assert_allclose(actual, expected, atol=1e-12)

    # L2 commutes with everything
    actual = L2_mat @ Lx_mat - Lx_mat @ L2_mat
    npt.assert_allclose(actual, 0, atol=1e-12)

    actual = L2_mat @ Ly_mat - Ly_mat @ L2_mat
    npt.assert_allclose(actual, 0, atol=1e-12)

    actual = L2_mat @ Lz_mat - Lz_mat @ L2_mat
    npt.assert_allclose(actual, 0, atol=1e-12)

    actual = L2_mat @ Lp_mat - Lp_mat @ L2_mat
    npt.assert_allclose(actual, 0, atol=1e-12)

    actual = L2_mat @ Lm_mat - Lm_mat @ L2_mat
    npt.assert_allclose(actual, 0, atol=1e-12)
    
@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("size", [4, 20])
def test_diagonalization(size, buffered):
    am = xyz2nlm.CreateAngularMomentumMatrices(size)

    if buffered:
        operator_buffer_size = am.basis_size
    else:
        operator_buffer_size = None

    Lx, Ly, Lz = am.get_linear_operators(operator_buffer_size)
    L2 = am.get_casimir()

    eye = np.eye(am.basis_size, dtype=complex)
    H = L2 @ eye + Lz @ eye
    e = np.linalg.eigvalsh(H)
    e = np.sort(e)

    expected = np.array([l*(l+1) + m \
                         for l in range(0 if (size % 2 == 0) else 1, size+1, 2) \
                         for m in range(-l, l+1)])

    npt.assert_allclose(e, expected, atol=1e-12)
