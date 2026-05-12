import numpy as np
import numpy.testing as npt
import pytest

import xyz2nlm
from xyz2nlm.qi_interface import exact_zero_array, is_zero, qi

@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("size", [4, 20])
def test_commutators(size, exact, buffered):
    am = xyz2nlm.create_angular_momentum_matrices(size, exact=exact)

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


def exact_eye(size):
    eye = exact_zero_array((size, size))
    for i in range(size):
        eye[i, i] = qi(1)
    return eye


def assert_exact_zero(array):
    assert all(is_zero(value) for value in array.flat)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4])
def test_exact_commutators(size):
    am = xyz2nlm.create_angular_momentum_matrices(size)

    Lx, Ly, Lz = am.get_linear_operators()
    Lp, Lm = am.get_ladder_operators()
    eye = exact_eye(am.basis_size)

    Lx_mat = Lx @ eye
    Ly_mat = Ly @ eye
    Lz_mat = Lz @ eye
    Lp_mat = Lp @ eye
    Lm_mat = Lm @ eye

    assert_exact_zero(Lx_mat @ Ly_mat - Ly_mat @ Lx_mat - Lz_mat * qi(1j))
    assert_exact_zero(Ly_mat @ Lz_mat - Lz_mat @ Ly_mat - Lx_mat * qi(1j))
    assert_exact_zero(Lz_mat @ Lx_mat - Lx_mat @ Lz_mat - Ly_mat * qi(1j))
    assert_exact_zero(Lz_mat @ Lp_mat - Lp_mat @ Lz_mat - Lp_mat)
    assert_exact_zero(Lz_mat @ Lm_mat - Lm_mat @ Lz_mat + Lm_mat)
    assert_exact_zero(Lp_mat @ Lm_mat - Lm_mat @ Lp_mat - Lz_mat * qi(2))
    
@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("size", [4, 20])
def test_diagonalization(size, exact, buffered):
    am = xyz2nlm.create_angular_momentum_matrices(size, exact=exact)

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
