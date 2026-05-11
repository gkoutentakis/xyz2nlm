Create the spherical harmonic oscillator basis from the Cartesian one.

By default, `xyz2nlm.SphericalHOBasis` now constructs coefficients exactly with
`QINumber` from `xyz2nlm.qi_algebra`. Dense shell blocks therefore use
`dtype=object`, and sparse state `non_zero_val` arrays contain exact algebraic
numbers rather than complex doubles.

Use explicit conversion helpers for numerical inspection:

```python
import xyz2nlm

basis = xyz2nlm.SphericalHOBasis(4)
basis.calculate_states(method="Serial")

complex_blocks = basis.as_complex_basis_states_views_n()
```

The legacy floating-point construction is still available with
`xyz2nlm.SphericalHOBasis(Nmax, exact=False)`. Exact mode is slower, but avoids
the roundoff accumulation that appears in high-shell ladder construction.
