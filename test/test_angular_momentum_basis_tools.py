import numpy as np
import numpy.testing as npt
import pytest
import inspect

import xyz2nlm._angular_momentum_basis_tools as amtools

def _test_AngularMomentumBasis(cls):
    indices = np.arange(cls.Nstates)

    j, m = cls.index2jm(indices)
    npt.assert_array_equal(cls.index_list[:, 0], j)
    npt.assert_array_equal(cls.index_list[:, 1], m)

    indices_roundtrip = cls.jm2index(j, m)
    npt.assert_array_equal(indices_roundtrip, indices)

def log_normalization(logger, jmax, jmax_real):
    name = inspect.currentframe().f_back.f_code.co_name
    logger(f"{name:<35}: jmax requested: {jmax!s:>4}, realized: {jmax_real!s:>4}")

@pytest.mark.parametrize("jmax", [4, 4.7, 5.5, 6.3, 20])
def test_FullAngularMomentumBasis(jmax, logger):

    cls = amtools.FullAngularMomentumBasis(jmax)

    jmax_real = np.floor(2 * jmax).astype(int) / 2.0

    log_normalization(logger, jmax, jmax_real)

    assert cls.jmax == jmax_real
    assert cls.index_list[-1, 0] == jmax_real
    assert cls.Nstates == (jmax_real + 1)*(2 * jmax_real + 1)
    _test_AngularMomentumBasis(cls)

@pytest.mark.parametrize("jmax", [4, 5.5, 6.3, 20])
def test_IntegerAngularMomentumBasis(jmax, logger):

    cls = amtools.IntegerAngularMomentumBasis(jmax)

    jmax_real = np.floor(jmax).astype(int)

    log_normalization(logger, jmax, jmax_real)

    assert cls.jmax == jmax_real
    assert cls.index_list[-1, 0] == jmax_real
    assert cls.Nstates == (jmax_real + 1)**2
    _test_AngularMomentumBasis(cls)

@pytest.mark.parametrize("jmax", [4, 5.5, 6.3, 20])
def test_EvenAngularMomentumBasis(jmax, logger):

    cls = amtools.EvenAngularMomentumBasis(jmax)

    jmax_real = np.floor(jmax).astype(int)
    jmax_real = jmax_real if (jmax_real % 2 == 0) else jmax_real-1

    log_normalization(logger, jmax, jmax_real)

    assert cls.jmax == jmax_real
    assert cls.index_list[-1, 0] == jmax_real
    assert cls.Nstates == ((jmax_real + 1) * (jmax_real + 2)) //2
    _test_AngularMomentumBasis(cls)

@pytest.mark.parametrize("jmax", [4, 5.5, 6.3, 20])
def test_OddAngularMomentumBasis(jmax, logger):

    cls = amtools.OddAngularMomentumBasis(jmax)

    jmax_real = np.floor(jmax).astype(int)
    jmax_real = jmax_real if (jmax_real % 2 == 1) else jmax_real-1

    log_normalization(logger, jmax, jmax_real)

    assert cls.jmax == jmax_real
    assert cls.index_list[-1, 0] == jmax_real
    assert cls.Nstates == ((jmax_real + 1) * (jmax_real + 2))//2
    _test_AngularMomentumBasis(cls)
