import numpy as np
import numpy.testing as npt
import pytest
import inspect

import xyz2nlm
from xyz2nlm.custom_dataclasses import TransientState
from xyz2nlm.direct_seed_state_calculation import create_seed_state

def get_norm(state):
    state_array = state.as_ndarray()
    return np.conjugate(state_array).T @ state_array

def get_expectation_values(operator, state):
    state_array = state.as_ndarray()
    return np.conjugate(state_array).T @ (operator @ state_array)

def get_expectation_value_and_variance(operator, state):
    expectation_value = get_expectation_values(operator, state)
    expectation_value_of_square = get_expectation_values(operator @ operator, state)
    variance = expectation_value_of_square - expectation_value**2
    return expectation_value, variance

def test_initial_state():
    shob = xyz2nlm.SphericalHOBasis(1)
    stat = shob._initial_state_even_sector()

    am = xyz2nlm.CreateAngularMomentumMatrices(0)
    Lx, Ly, Lz = am.get_linear_operators()
    L2 = am.get_casimir()

    L2_av, L2_var = get_expectation_value_and_variance(L2, stat)
    assert L2_av == 0
    assert L2_var == 0

    Lz_av, Lz_var = get_expectation_value_and_variance(Lz, stat)
    assert Lz_av == 0
    assert Lz_var == 0
    
def test_s2_operator():
    shob = xyz2nlm.SphericalHOBasis(10)
    stat = shob._initial_state_even_sector()

    for i in range(5):
        stat= shob.s2_operator(stat)

        assert stat.n == 2 * (i + 1)
        assert stat.l == 0
        assert stat.m == 0

        am = xyz2nlm.CreateAngularMomentumMatrices(stat.n)
        Lx, Ly, Lz = am.get_linear_operators()
        L2 = am.get_casimir()

        norm = get_norm(stat)
        npt.assert_allclose(norm,  1, atol=1e-12)

        L2_av, L2_var = get_expectation_value_and_variance(L2, stat)
        npt.assert_allclose(L2_av,  stat.l * (stat.l + 1), atol=1e-12)
        npt.assert_allclose(L2_var, 0, atol=1e-12)

        Lz_av, Lz_var = get_expectation_value_and_variance(Lz, stat)
        npt.assert_allclose(Lz_av,  stat.m, atol=1e-12)
        npt.assert_allclose(Lz_var, 0, atol=1e-12)

def test_r22_operator():
    shob = xyz2nlm.SphericalHOBasis(10)
    stat = shob._initial_state_even_sector()

    for i in range(5):
        stat= shob.r22_operator(stat)

        assert stat.n == 2 * (i + 1)
        assert stat.l == 2 * (i + 1)
        assert stat.m == 2 * (i + 1)

        am = xyz2nlm.CreateAngularMomentumMatrices(stat.n)
        Lx, Ly, Lz = am.get_linear_operators()
        L2 = am.get_casimir()
        
        norm = get_norm(stat)
        npt.assert_allclose(norm,  1, atol=1e-12)

        L2_av, L2_var = get_expectation_value_and_variance(L2, stat)
        npt.assert_allclose(L2_av,  stat.l * (stat.l + 1), atol=1e-12)
        npt.assert_allclose(L2_var, 0, atol=1e-11)

        Lz_av, Lz_var = get_expectation_value_and_variance(Lz, stat)
        npt.assert_allclose(Lz_av,  stat.m, atol=1e-12)
        npt.assert_allclose(Lz_var, 0, atol=1e-12)

def test_Lm_operator():
    shob = xyz2nlm.SphericalHOBasis(10)
    stat = shob._initial_state_even_sector()

    for i in range(5):
        stat= shob.r22_operator(stat)

        assert stat.n == 2 * (i + 1)
        assert stat.l == 2 * (i + 1)
        assert stat.m == 2 * (i + 1)

        am = xyz2nlm.CreateAngularMomentumMatrices(stat.n)
        Lx, Ly, Lz = am.get_linear_operators()
        Lp, Lm = am.get_ladder_operators()
        L2 = am.get_casimir()

        stat_check = TransientState.from_ndarray(stat.n, Lm @ stat.as_ndarray())\
                                   .scale(1/np.sqrt((stat.l+stat.m)*(stat.l-stat.m+1)))
        
        norm = get_norm(stat_check)
        npt.assert_allclose(norm,  1, atol=1e-12)

        L2_av, L2_var = get_expectation_value_and_variance(L2, stat_check)
        npt.assert_allclose(L2_av,  stat.l * (stat.l + 1), atol=1e-12)
        npt.assert_allclose(L2_var, 0, atol=1e-11)

        Lz_av, Lz_var = get_expectation_value_and_variance(Lz, stat_check)
        npt.assert_allclose(Lz_av,  stat.m - 1, atol=1e-12)
        npt.assert_allclose(Lz_var, 0, atol=1e-12)

@pytest.mark.parametrize("N,L", [(2, 0), (2, 2),
                                 (10, 0), (10, 4), (10, 10),
                                 (100, 0), (100, 50), (100, 100)])
def test_direct_generation(N, L):
        
    stat = create_seed_state(N, L)
    stat_array = stat.as_ndarray()

    norm = get_norm(stat)
    npt.assert_allclose(norm,  1, atol=1e-12)

    am = xyz2nlm.CreateAngularMomentumMatrices(stat.n)
    Lx, Ly, Lz = am.get_linear_operators()
    Lp, Lm = am.get_ladder_operators()
    L2 = am.get_casimir()

    L2_av, L2_var = get_expectation_value_and_variance(L2, stat)
    npt.assert_allclose(L2_av,  stat.l * (stat.l + 1), atol=1e-12)
    npt.assert_allclose(L2_var, 0, atol=1e-11)

    Lz_av, Lz_var = get_expectation_value_and_variance(Lz, stat)
    npt.assert_allclose(Lz_av,  stat.m, atol=1e-12)
    npt.assert_allclose(Lz_var, 0, atol=1e-12)
