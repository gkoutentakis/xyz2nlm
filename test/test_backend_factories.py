import numpy as np

import xyz2nlm
from xyz2nlm import create_angular_momentum_matrices, create_spherical_basis
from xyz2nlm.custom_dataclasses import (
    SphericalHOBasisStateFactory,
    SphericalHOBasisStateNP,
    SphericalHOBasisStateQI,
    TransientStateFactory,
    TransientStateNP,
    TransientStateQI,
)
from xyz2nlm.qi_interface import exact_zero_array, qi


def test_create_angular_momentum_factory_selects_backend():
    assert isinstance(
        xyz2nlm.create_angular_momentum_matrices(2),
        xyz2nlm.AngularMomentumMatricesQI,
    )
    assert isinstance(
        xyz2nlm.create_angular_momentum_matrices(2, exact=True),
        xyz2nlm.AngularMomentumMatricesQI,
    )
    assert isinstance(
        xyz2nlm.create_angular_momentum_matrices(2, exact=False),
        xyz2nlm.AngularMomentumMatricesNP,
    )


def test_create_angular_momentum_implementations_match_protocol():
    assert isinstance(
        xyz2nlm.create_angular_momentum_matrices(2),
        xyz2nlm.AngularMomentumMatrices,
    )
    assert isinstance(
        xyz2nlm.create_angular_momentum_matrices(2, exact=False),
        xyz2nlm.AngularMomentumMatrices,
    )
    assert isinstance(
        xyz2nlm.AngularMomentumMatricesQI(2),
        xyz2nlm.AngularMomentumMatrices,
    )
    assert isinstance(
        xyz2nlm.AngularMomentumMatricesNP(2),
        xyz2nlm.AngularMomentumMatrices,
    )


def test_spherical_basis_factory_selects_backend():
    assert isinstance(xyz2nlm.create_spherical_ho_basis(2, calculate=False), xyz2nlm.SphericalHOBasisQI)
    assert isinstance(xyz2nlm.create_spherical_ho_basis(2, exact=True, calculate=False), xyz2nlm.SphericalHOBasisQI)
    assert isinstance(xyz2nlm.create_spherical_ho_basis(2, exact=False, calculate=False), xyz2nlm.SphericalHOBasisNP)


def test_spherical_basis_implementations_match_protocol():
    assert isinstance(xyz2nlm.create_spherical_ho_basis(2, calculate=False), xyz2nlm.SphericalHOBasis)
    assert isinstance(
        xyz2nlm.create_spherical_ho_basis(2, exact=False, calculate=False),
        xyz2nlm.SphericalHOBasis,
    )
    assert isinstance(xyz2nlm.SphericalHOBasisQI(2), xyz2nlm.SphericalHOBasis)
    assert isinstance(xyz2nlm.SphericalHOBasisNP(2), xyz2nlm.SphericalHOBasis)


def test_spherical_basis_uses_backend_state_classes():
    exact_basis = xyz2nlm.create_spherical_ho_basis(2, calculate=False)
    complex_basis = xyz2nlm.create_spherical_ho_basis(2, exact=False, calculate=False)

    assert isinstance(exact_basis._initial_state_even_sector(), SphericalHOBasisStateQI)
    assert isinstance(exact_basis._initial_state_odd_sector(), SphericalHOBasisStateQI)
    assert isinstance(complex_basis._initial_state_even_sector(), SphericalHOBasisStateNP)
    assert isinstance(complex_basis._initial_state_odd_sector(), SphericalHOBasisStateNP)


def test_transient_state_factory_selects_backend_from_values():
    complex_state = TransientStateFactory(
        n=2,
        non_zero_ind=np.array([0]),
        non_zero_val=np.array([1 + 1j]),
    )
    exact_state = TransientStateFactory(
        n=2,
        non_zero_ind=np.array([0]),
        non_zero_val=np.array([qi(1) + qi(1j)], dtype=object),
    )

    assert isinstance(complex_state, TransientStateNP)
    assert isinstance(exact_state, TransientStateQI)
    assert isinstance(complex_state, xyz2nlm.TransientState)
    assert isinstance(exact_state, xyz2nlm.TransientState)
    assert isinstance(
        TransientStateNP(
            n=2,
            non_zero_ind=np.array([0]),
            non_zero_val=np.array([1 + 1j]),
        ),
        xyz2nlm.TransientState,
    )
    assert isinstance(
        TransientStateQI(
            n=2,
            non_zero_ind=np.array([0]),
            non_zero_val=np.array([qi(1) + qi(1j)], dtype=object),
        ),
        xyz2nlm.TransientState,
    )


def test_transient_state_from_ndarray_factory_selects_backend():
    complex_dense = np.zeros((6,), dtype=complex)
    complex_dense[0] = 1 + 1j

    exact_dense = exact_zero_array((6,))
    exact_dense[0] = qi(1) + qi(1j)

    assert isinstance(TransientStateFactory.from_ndarray(2, complex_dense), TransientStateNP)
    assert isinstance(TransientStateFactory.from_ndarray(2, exact_dense), TransientStateQI)


def test_spherical_ho_basis_state_factory_selects_backend_from_values():
    complex_state = SphericalHOBasisStateFactory(
        n=2,
        l=0,
        m=0,
        non_zero_ind=np.array([0]),
        non_zero_val=np.array([1 + 1j]),
    )
    exact_state = SphericalHOBasisStateFactory(
        n=2,
        l=0,
        m=0,
        non_zero_ind=np.array([0]),
        non_zero_val=np.array([qi(1) + qi(1j)], dtype=object),
    )

    assert isinstance(complex_state, SphericalHOBasisStateNP)
    assert isinstance(exact_state, SphericalHOBasisStateQI)
    assert isinstance(complex_state, xyz2nlm.SphericalHOBasisState)
    assert isinstance(exact_state, xyz2nlm.SphericalHOBasisState)
    assert isinstance(
        SphericalHOBasisStateNP(
            n=2,
            l=0,
            m=0,
            non_zero_ind=np.array([0]),
            non_zero_val=np.array([1 + 1j]),
        ),
        xyz2nlm.SphericalHOBasisState,
    )
    assert isinstance(
        SphericalHOBasisStateQI(
            n=2,
            l=0,
            m=0,
            non_zero_ind=np.array([0]),
            non_zero_val=np.array([qi(1) + qi(1j)], dtype=object),
        ),
        xyz2nlm.SphericalHOBasisState,
    )


def test_public_api_exposes_protocols_not_base_abcs():
    public_names = set(xyz2nlm.__all__)

    assert "AngularMomentumMatricesBase" not in public_names
    assert "SphericalHOBasisBase" not in public_names
    assert not hasattr(xyz2nlm, "AngularMomentumMatricesBase")
    assert not hasattr(xyz2nlm, "SphericalHOBasisBase")
    assert not hasattr(create_angular_momentum_matrices, "AngularMomentumMatricesBase")
    assert not hasattr(create_spherical_basis, "SphericalHOBasisBase")

    assert "AngularMomentumMatrices" in public_names
    assert "SphericalHOBasis" in public_names
    assert "SphericalHOBasisState" in public_names
    assert "TransientState" in public_names
