import numpy as np
import numpy.testing as npt
import pytest

import xyz2nlm
from xyz2nlm.angular_momentum_basis_tools import EvenAngularMomentumBasis, OddAngularMomentumBasis
from xyz2nlm.qi_interface import exact_inner_product, is_zero, qi, qi_sqrt_int


def _basis_index(n):
    if n % 2 == 0:
        return EvenAngularMomentumBasis(n)
    return OddAngularMomentumBasis(n)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 5])
def test_spherical_basis_exact_small_shells(size):
    shob = xyz2nlm.create_spherical_ho_basis(size, method="Serial")

    for n in range(size + 1):
        bas = shob.basis_states_views_n[n]
        assert bas.shape == (((n + 1) * (n + 2)) // 2,) * 2

        indices = _basis_index(n).index_list
        am = xyz2nlm.create_angular_momentum_matrices(n, compatible_basis=shob)
        _Lx, _Ly, Lz = am.get_linear_operators()

        for i, (_ell, m) in enumerate(indices):
            assert exact_inner_product(bas[:, i], bas[:, i]) == 1
            lz_action = Lz @ bas[:, i]
            assert all(is_zero(actual - qi(int(m)) * expected) for actual, expected in zip(lz_action, bas[:, i]))


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 5])
def test_spherical_basis_complex_small_shells(size):
    shob = xyz2nlm.create_spherical_ho_basis(size, exact=False, method="Serial")

    for n in range(size + 1):
        bas = shob.basis_states_views_n[n]
        assert bas.shape == (((n + 1) * (n + 2)) // 2,) * 2

        indices = _basis_index(n).index_list
        am = xyz2nlm.create_angular_momentum_matrices(n, compatible_basis=shob)
        _Lx, _Ly, Lz = am.get_linear_operators()

        for i, (_ell, m) in enumerate(indices):
            npt.assert_allclose(np.conjugate(bas[:, i]).T @ bas[:, i], 1, atol=1e-12)
            npt.assert_allclose(Lz @ bas[:, i], m * bas[:, i], atol=1e-12)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 5, 20])
def test_spherical_basis_exact_lowering(size):
    shob = xyz2nlm.create_spherical_ho_basis(size, method="Serial")

    for n in range(size + 1):
        bas = shob.basis_states_views_n[n]
        indices = _basis_index(n)
        am = xyz2nlm.create_angular_momentum_matrices(n)
        _Lp, Lm = am.get_ladder_operators()

        for ell, m in indices.index_list:
            if m <= -ell:
                continue
            state = bas[:, indices.jm2index(ell, m)]
            expected = bas[:, indices.jm2index(ell, m - 1)]
            factor = qi_sqrt_int((ell + m) * (ell - m + 1))
            actual = Lm @ state
            assert all(is_zero(a - factor * b) for a, b in zip(actual, expected))


# The exact implementation is tested at size 20 above. The complex-double path
# keeps the smaller range here because repeated laddering accumulates roundoff.
@pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 5])
def test_spherical_basis_complex_lowering(size):
    shob = xyz2nlm.create_spherical_ho_basis(size, exact=False, method="Serial")

    for n in range(size + 1):
        bas = shob.basis_states_views_n[n]
        indices = _basis_index(n)
        am = xyz2nlm.create_angular_momentum_matrices(n, exact=False)
        _Lp, Lm = am.get_ladder_operators()

        for ell, m in indices.index_list:
            if m <= -ell:
                continue
            state = bas[:, indices.jm2index(ell, m)]
            expected = bas[:, indices.jm2index(ell, m - 1)]
            factor = np.sqrt((ell + m) * (ell - m + 1))
            npt.assert_allclose(Lm @ state, factor * expected, atol=1e-12)


@pytest.mark.parametrize("size", [4, 20])
def test_spherical_basis_complex_compatibility(size):
    shob_exact = xyz2nlm.create_spherical_ho_basis(size, method="Serial")

    shob_np = xyz2nlm.create_spherical_ho_basis(size, exact=False, method="Serial")

    exact_complex_views = shob_exact.as_complex_basis_states_views_n()
    for n in range(size + 1):
        exact_complex = exact_complex_views[n]
        np_basis = shob_np.basis_states_views_n[n]

        npt.assert_allclose(exact_complex, np_basis, atol=1e-8, rtol=0, err_msg=f"state {n}")

        am = xyz2nlm.create_angular_momentum_matrices(n, exact=False)
        _Lx, _Ly, Lz = am.get_linear_operators()
        L2 = am.get_casimir()

        indices = _basis_index(n).index_list
        ovrl_actual = np.conjugate(exact_complex).T @ exact_complex
        L2_av_actual = np.conjugate(exact_complex).T @ (L2 @ exact_complex)
        Lz_av_actual = np.conjugate(exact_complex).T @ (Lz @ exact_complex)

        ovrl_expected = np.eye(ovrl_actual.shape[0])
        L2_expected = indices[:, 0].astype(complex)
        L2_expected = L2_expected * (L2_expected + 1)
        Lz_expected = indices[:, 1].astype(complex)

        npt.assert_allclose(ovrl_actual, ovrl_expected, atol=1e-8, rtol=0, err_msg=f"state {n}")
        npt.assert_allclose(np.diag(L2_av_actual), L2_expected, atol=1e-8, rtol=0, err_msg=f"state {n}")
        npt.assert_allclose(np.diag(Lz_av_actual), Lz_expected, atol=1e-8, rtol=0, err_msg=f"state {n}")
