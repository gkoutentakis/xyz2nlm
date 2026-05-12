import numpy as np
import numpy.testing as npt
import pytest
import inspect

import xyz2nlm
from xyz2nlm.custom_dataclasses import TransientStateNP, TransientStateQI
from xyz2nlm.direct_seed_state_calculation import create_seed_state
from xyz2nlm.qi_interface import exact_inner_product, exact_norm, is_zero, qi, qi_sqrt_int

def get_norm(state):
    return exact_norm(state)

def get_expectation_values(operator, state):
    state_array = state.as_ndarray()
    return exact_inner_product(state_array, operator @ state_array)

def get_expectation_value_and_variance(operator, state):
    expectation_value = get_expectation_values(operator, state)
    state_array = state.as_ndarray()
    expectation_value_of_square = exact_inner_product(
        state_array,
        operator @ (operator @ state_array),
    )
    variance = expectation_value_of_square - expectation_value**2
    return expectation_value, variance

def get_norm_complex(state):
    state_array = state.as_ndarray()
    return np.conjugate(state_array).T @ state_array

def get_expectation_values_complex(operator, state):
    state_array = state.as_ndarray()
    return np.conjugate(state_array).T @ (operator @ state_array)

def get_expectation_value_and_variance_complex(operator, state):
    expectation_value = get_expectation_values_complex(operator, state)
    expectation_value_of_square = get_expectation_values_complex(operator @ operator, state)
    variance = expectation_value_of_square - expectation_value**2
    return expectation_value, variance

def test_initial_state():
    shob = xyz2nlm.create_spherical_ho_basis(1, calculate=False)
    stat = shob._initial_state_even_sector()

    am = xyz2nlm.create_angular_momentum_matrices(0)
    Lx, Ly, Lz = am.get_linear_operators()
    L2 = am.get_casimir()

    L2_av, L2_var = get_expectation_value_and_variance(L2, stat)
    assert L2_av == 0
    assert L2_var == 0

    Lz_av, Lz_var = get_expectation_value_and_variance(Lz, stat)
    assert Lz_av == 0
    assert Lz_var == 0

def test_initial_state_complex():
    shob = xyz2nlm.create_spherical_ho_basis(1, exact=False, calculate=False)
    stat = shob._initial_state_even_sector()

    am = xyz2nlm.create_angular_momentum_matrices(0, exact=False)
    _Lx, _Ly, Lz = am.get_linear_operators()
    L2 = am.get_casimir()

    L2_av, L2_var = get_expectation_value_and_variance_complex(L2, stat)
    npt.assert_allclose(L2_av, 0, atol=1e-12)
    npt.assert_allclose(L2_var, 0, atol=1e-12)

    Lz_av, Lz_var = get_expectation_value_and_variance_complex(Lz, stat)
    npt.assert_allclose(Lz_av, 0, atol=1e-12)
    npt.assert_allclose(Lz_var, 0, atol=1e-12)
    
def test_s2_operator():
    shob = xyz2nlm.create_spherical_ho_basis(10, calculate=False)
    stat = shob._initial_state_even_sector()

    for i in range(5):
        stat= shob.s2_operator(stat)

        assert stat.n == 2 * (i + 1)
        assert stat.l == 0
        assert stat.m == 0

        am = xyz2nlm.create_angular_momentum_matrices(stat.n)
        Lx, Ly, Lz = am.get_linear_operators()
        L2 = am.get_casimir()

        norm = get_norm(stat)
        assert norm == 1

        L2_av, L2_var = get_expectation_value_and_variance(L2, stat)
        assert L2_av == stat.l * (stat.l + 1)
        assert L2_var == 0

        Lz_av, Lz_var = get_expectation_value_and_variance(Lz, stat)
        assert Lz_av == stat.m
        assert Lz_var == 0

def test_s2_operator_complex():
    shob = xyz2nlm.create_spherical_ho_basis(10, exact=False, calculate=False)
    stat = shob._initial_state_even_sector()

    for i in range(5):
        stat = shob.s2_operator(stat)

        assert stat.n == 2 * (i + 1)
        assert stat.l == 0
        assert stat.m == 0

        am = xyz2nlm.create_angular_momentum_matrices(stat.n, exact=False)
        _Lx, _Ly, Lz = am.get_linear_operators()
        L2 = am.get_casimir()

        npt.assert_allclose(get_norm_complex(stat), 1, atol=1e-12)

        L2_av, L2_var = get_expectation_value_and_variance_complex(L2, stat)
        npt.assert_allclose(L2_av, stat.l * (stat.l + 1), atol=1e-12)
        npt.assert_allclose(L2_var, 0, atol=1e-12)

        Lz_av, Lz_var = get_expectation_value_and_variance_complex(Lz, stat)
        npt.assert_allclose(Lz_av, stat.m, atol=1e-12)
        npt.assert_allclose(Lz_var, 0, atol=1e-12)

def test_r22_operator():
    shob = xyz2nlm.create_spherical_ho_basis(10, calculate=False)
    stat = shob._initial_state_even_sector()

    for i in range(5):
        stat= shob.r22_operator(stat)

        assert stat.n == 2 * (i + 1)
        assert stat.l == 2 * (i + 1)
        assert stat.m == 2 * (i + 1)

        am = xyz2nlm.create_angular_momentum_matrices(stat.n)
        Lx, Ly, Lz = am.get_linear_operators()
        L2 = am.get_casimir()
        
        norm = get_norm(stat)
        assert norm == 1

        L2_av, L2_var = get_expectation_value_and_variance(L2, stat)
        assert L2_av == stat.l * (stat.l + 1)
        assert L2_var == 0

        Lz_av, Lz_var = get_expectation_value_and_variance(Lz, stat)
        assert Lz_av == stat.m
        assert Lz_var == 0

def test_r22_operator_complex():
    shob = xyz2nlm.create_spherical_ho_basis(10, exact=False, calculate=False)
    stat = shob._initial_state_even_sector()

    for i in range(5):
        stat = shob.r22_operator(stat)

        assert stat.n == 2 * (i + 1)
        assert stat.l == 2 * (i + 1)
        assert stat.m == 2 * (i + 1)

        am = xyz2nlm.create_angular_momentum_matrices(stat.n, exact=False)
        _Lx, _Ly, Lz = am.get_linear_operators()
        L2 = am.get_casimir()
        
        npt.assert_allclose(get_norm_complex(stat), 1, atol=1e-12)

        L2_av, L2_var = get_expectation_value_and_variance_complex(L2, stat)
        npt.assert_allclose(L2_av, stat.l * (stat.l + 1), atol=1e-12)
        npt.assert_allclose(L2_var, 0, atol=1e-11)

        Lz_av, Lz_var = get_expectation_value_and_variance_complex(Lz, stat)
        npt.assert_allclose(Lz_av, stat.m, atol=1e-12)
        npt.assert_allclose(Lz_var, 0, atol=1e-12)

def test_Lm_operator():
    shob = xyz2nlm.create_spherical_ho_basis(10, calculate=False)
    stat = shob._initial_state_even_sector()

    for i in range(5):
        stat= shob.r22_operator(stat)

        assert stat.n == 2 * (i + 1)
        assert stat.l == 2 * (i + 1)
        assert stat.m == 2 * (i + 1)

        am = xyz2nlm.create_angular_momentum_matrices(stat.n)
        Lx, Ly, Lz = am.get_linear_operators()
        Lp, Lm = am.get_ladder_operators()
        L2 = am.get_casimir()

        stat_check = TransientStateQI.from_ndarray(stat.n, Lm @ stat.as_ndarray())\
                                     .scale(qi(1) / qi_sqrt_int((stat.l+stat.m)*(stat.l-stat.m+1)))
        
        norm = get_norm(stat_check)
        assert norm == 1

        L2_av, L2_var = get_expectation_value_and_variance(L2, stat_check)
        assert L2_av == stat.l * (stat.l + 1)
        assert L2_var == 0

        Lz_av, Lz_var = get_expectation_value_and_variance(Lz, stat_check)
        assert Lz_av == stat.m - 1
        assert Lz_var == 0

def test_Lm_operator_complex():
    shob = xyz2nlm.create_spherical_ho_basis(10, exact=False, calculate=False)
    stat = shob._initial_state_even_sector()

    for i in range(5):
        stat = shob.r22_operator(stat)

        assert stat.n == 2 * (i + 1)
        assert stat.l == 2 * (i + 1)
        assert stat.m == 2 * (i + 1)

        am = xyz2nlm.create_angular_momentum_matrices(stat.n, exact=False)
        _Lx, _Ly, Lz = am.get_linear_operators()
        _Lp, Lm = am.get_ladder_operators()
        L2 = am.get_casimir()

        stat_check = TransientStateNP.from_ndarray(stat.n, Lm @ stat.as_ndarray())\
                                     .scale(1 / np.sqrt((stat.l+stat.m)*(stat.l-stat.m+1)))
        
        npt.assert_allclose(get_norm_complex(stat_check), 1, atol=1e-12)

        L2_av, L2_var = get_expectation_value_and_variance_complex(L2, stat_check)
        npt.assert_allclose(L2_av, stat.l * (stat.l + 1), atol=1e-12)
        npt.assert_allclose(L2_var, 0, atol=1e-11)

        Lz_av, Lz_var = get_expectation_value_and_variance_complex(Lz, stat_check)
        npt.assert_allclose(Lz_av, stat.m - 1, atol=1e-12)
        npt.assert_allclose(Lz_var, 0, atol=1e-12)

@pytest.mark.parametrize("N,L", [(2, 0), (2, 2),
                                 (10, 0), (10, 4), (10, 10),
                                 (100, 0), (100, 50), (100, 100)])
def test_direct_generation(N, L):
        
    stat = create_seed_state(N, L)

    norm = get_norm(stat)
    assert norm == 1

    am = xyz2nlm.create_angular_momentum_matrices(stat.n)
    Lx, Ly, Lz = am.get_linear_operators()
    Lp, Lm = am.get_ladder_operators()
    L2 = am.get_casimir()

    L2_av, L2_var = get_expectation_value_and_variance(L2, stat)
    assert L2_av == stat.l * (stat.l + 1)
    assert L2_var == 0

    Lz_av, Lz_var = get_expectation_value_and_variance(Lz, stat)
    assert Lz_av == stat.m
    assert Lz_var == 0

    Lp_action = Lp @ stat.as_ndarray()
    assert all(is_zero(value) for value in Lp_action)

# The exact path above covers larger N. The complex-double path keeps this
# reduced historical range because larger seeds are roundoff-limited.
@pytest.mark.parametrize("N,L", [(2, 0), (2, 2),
                                 (10, 0), (10, 4)])
def test_direct_generation_complex(N, L):
    stat = create_seed_state(N, L, exact=False)

    npt.assert_allclose(get_norm_complex(stat), 1, atol=1e-12)

    am = xyz2nlm.create_angular_momentum_matrices(stat.n, exact=False)
    _Lx, _Ly, Lz = am.get_linear_operators()
    Lp, _Lm = am.get_ladder_operators()
    L2 = am.get_casimir()

    L2_av, L2_var = get_expectation_value_and_variance_complex(L2, stat)
    npt.assert_allclose(L2_av, stat.l * (stat.l + 1), atol=1e-12)
    npt.assert_allclose(L2_var, 0, atol=1e-11)

    Lz_av, Lz_var = get_expectation_value_and_variance_complex(Lz, stat)
    npt.assert_allclose(Lz_av, stat.m, atol=1e-12)
    npt.assert_allclose(Lz_var, 0, atol=1e-12)

    npt.assert_allclose(Lp @ stat.as_ndarray(), 0, atol=1e-12)
