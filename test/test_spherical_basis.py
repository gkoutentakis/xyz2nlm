import numpy as np
import numpy.testing as npt
import pytest
import inspect

import xyz2nlm
from xyz2nlm._angular_momentum_basis_tools import EvenAngularMomentumBasis, OddAngularMomentumBasis

@pytest.mark.parametrize("size", [4, 20, 40])
def test_spherical_basis(size):
    shob = xyz2nlm.SphericalHOBasis(size)
    shob.calculate_states()

    for n in range(size+1):
        am = xyz2nlm.CreateAngularMomentumMatrices(n)
        Lx, Ly, Lz = am.get_linear_operators()
        L2 = am.get_casimir()

        bas = shob.basis_states_views_n[n]

        if n%2==0:
            indices = EvenAngularMomentumBasis(n).index_list
        else:
            indices = OddAngularMomentumBasis(n).index_list

        ovrl_actual = np.conjugate(bas).T @ bas
        L2_av_actual = np.conjugate(bas).T @ (L2 @ bas)
        Lz_av_actual = np.conjugate(bas).T @ (Lz @ bas)

        ovrl_expected = np.eye(ovrl_actual.shape[0])
        L2_expected = indices[:, 0].astype(complex)
        L2_expected = L2_expected * (L2_expected + 1)
        Lz_expected = indices[:, 1].astype(complex)

        npt.assert_allclose(ovrl_actual, ovrl_expected, atol=1e-6, rtol=0, err_msg=f"state {n}")
        npt.assert_allclose(np.diag(L2_av_actual), L2_expected, atol=1e-6, rtol=0, err_msg=f"state {n}")
        npt.assert_allclose(np.diag(Lz_av_actual), Lz_expected, atol=1e-6, rtol=0, err_msg=f"state {n}")

