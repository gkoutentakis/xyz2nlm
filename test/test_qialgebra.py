import numpy as np
import numpy.testing as npt
import pytest
import inspect

from xyz2nlm.qi_algebra import QINumber, MathDomainError

qi = lambda n: QINumber(n=n)
qs = lambda n: QINumber(n=n, sqrt=True)

tests_equal = [
    (qs(8),                         qi(2)*qs(2)),
    (qs(12),                        qi(2)*qs(3)),
    (qs(18),                        qi(3)*qs(2)),
    (qs(0),                         qi(0)),
    (qs(1),                         qi(1)),
    (qi(3+1j)/qi(2)*qs(12),         qi(3+1j)*qs(3)),
    (qi(2+1j)/qi(3)*qs(18),         qi(2+1j)*qs(2)),
    (qi(0)*qs(2)+qi(5)*qs(5),       qi(5)*qs(5)),
    (qi(2)*qs(3)+qi(5)*qs(3),       qi(7)*qs(3)),
    (qi(1+1j)*qs(2)+qi(3-2j)*qs(2), qi(4-1j)*qs(2)),
    (qi(2)/qi(-3)*qs(5),            qi(-2)/qi(3)*qs(5)),
    (qs(2)+qs(8),                   qi(3)*qs(2)),
    (qs(2)*qs(3),                   qs(6)),
    (qs(12)+qs(27),                 qi(5)*qs(3)),
    (qi(1+1j)/qi(2)*qs(4),          qi(1+1j)),
    (qi(1j)*qs(2)+qi(1j)*qs(8),     qi(3j)*qs(2)),
    (qi(0)*qs(2)+qi(0)*qs(8),       qi(0)),
]

tests_not_equal = [
    (qs(2)+qs(3),                   qi(2)*qs(5))
]

tests_addition = [
    (qs(2)+qi(2)*qs(3),                   qi(-3)*qs(2)+qi(5j)*qs(3),
     qi(-2)*qs(2)+qi(2+5j)*qs(3)),
    (qi(1+1j),                            qi(2-3j),
     qi(3-2j)),
    (qi(1)/qi(2)*qs(2)+qi(1)/qi(3)*qs(3), qi(5)/qi(6)*qs(2)-qi(1)/qi(3)*qs(3),
     qi(4)/qi(3)*qs(2)),
]

tests_subtraction = [
    (qs(2)+qi(2)*qs(3),                   qi(3)*qs(2)-qi(5j)*qs(3),
     -qi(2)*qs(2)+qi(2+5j)*qs(3))
    
]

tests_multiplication = [
    (qs(2),                            qs(3),                            qs(6)),
    (qs(2),                            qs(8),                            qi(4)),
    (qs(12),                           qs(27),                           qi(18)),
    (qi(1j)*qs(2),                     qi(1j)*qs(2),                     qi(-2)),
    ((qi(1)/qi(2)+qi(1j)/qi(2))*qs(2), (qi(1)/qi(2)+qi(1j)/qi(2))*qs(2), qi(1j)),
    (qs(2)+qs(3),                      qs(2)+qs(3),                      qi(5) + qi(2)*qs(6)),
    (qs(2)+qs(3),                      qs(2)-qs(3),                      qi(-1)),
    (qi(1)+qs(2),                      qi(1)-qs(2),                      qi(-1)),
    (qs(2)+qi(1j)*qs(3),               qs(2)+qi(1j)*qs(3),               qi(-1)+qi(2j)*qs(6))
]

tests_inversion = [
    (qs(2),                         qs(2)/qi(2)),
    (qi(2)*qs(3),                   qs(3)/qi(6)),
    (qi(1)+qi(1j),                  (qi(1)-qi(1j))/qi(2)),
    (qi(2)+qi(1j)*qs(3),            (qi(2)-qi(1j)*qs(3))/qi(7)),
    (qs(2)+qi(1j)*qs(3),            (qs(2)-qi(1j)*qs(3))/qi(5)),
    (qi(1)+qs(2),                   qs(2)-qi(1)),
    (qs(2)+qs(3),                   qs(3)-qs(2)),
    (qi(1)+qs(2)+qs(3),             (qi(2)+qs(2)-qs(6))/qi(4)),
    (qi(1)+qs(2)+qs(3)+qs(5)+qs(7), qi(263)/qi(2018)*qs(2)\
                                    + qi(289)/qi(1009)*qs(1)\
                                    + qi(-45)/qi(2018)*qs(15)\
                                    + qi(-133)/qi(6054)*qs(30)\
                                    + qi(121)/qi(2018)*qs(10)\
                                    + qi(179)/qi(1009)*qs(5)\
                                    + qi(-326)/qi(3027)*qs(3)\
                                    + qi(-391)/qi(2018)*qs(6)\
                                    + qi(277)/qi(6054)*qs(21)\
                                    + qi(81)/qi(2018)*qs(42)\
                                    + qi(-39)/qi(2018)*qs(14)\
                                    + qi(-16)/qi(1009)*qs(7)\
                                    + qi(77)/qi(3027)*qs(210)\
                                    + qi(-1)/qi(2018)*qs(105)\
                                    + qi(-149)/qi(2018)*qs(35)\
                                    + qi(-87)/qi(2018)*qs(70))
]

tests_division = [
    (qs(6)+qi(2),        qs(2),              qs(3)+qs(2)),
    (qs(3)+qs(2),        qs(3)-qs(2),        qi(5)+qi(2)*qs(6)),
    (qs(2)+qi(1j)*qs(3), qs(2)-qi(1j)*qs(3), (-qi(1)+qi(2j)*qs(6))/qi(5)),
    (qi(1)+qs(2),        qs(1)-qs(2),        -qi(3)-qi(2)*qs(2))
]

tests_power = [
    (qs(2)+qs(3),   2, qi(5)+qi(2)*qs(6)),
    (qs(2)+qs(3),   3, qi(11)*qs(2)+qi(9)*qs(3)),
    (qi(1)+qs(2),   2, qi(3)+qi(2)*qs(2)),
    (qi(1)+qs(2),  -1, qs(2)-qi(1)),
    (qi(1)+qs(2),  -2, qi(3)-qi(2)*qs(2)),
    (qi(1j)*qs(2),  2, -qi(2)),
    (qi(1j),        4, qi(1)),
    (-qs(2),        3, -qi(2)*qs(2)),
]
@pytest.mark.parametrize("a, b", tests_equal)
def test_normalization(a, b):
    assert a == b

@pytest.mark.parametrize("a, b", tests_not_equal)
def test_not_equal(a, b):
    assert a != 0

@pytest.mark.parametrize("a, b", tests_equal)
def test_zero_equality(a, b):
    assert a - b == 0

@pytest.mark.parametrize("a, b, c", tests_addition)
def test_qi_sum(a, b, c):
    assert a + b == c

@pytest.mark.parametrize("a, b, c", tests_subtraction)
def test_qi_minus(a, b, c):
    assert a - b == c

@pytest.mark.parametrize("a, b, c", tests_addition)
def test_qi_addition_subtraction_properties(a, b, c):
    assert a + 0 == a
    assert a + b == b + a
    assert (a + b) + c == a + (b + c)
    assert a - a == 0
    assert a - 0 == a
    assert 0 - a == -a
    assert a - b == a + (- b)

@pytest.mark.parametrize("a, b, c", tests_multiplication)
def test_qi_times(a, b, c):
    assert a * b == c

@pytest.mark.parametrize("a, b, c", tests_addition)
def test_qi_addition_multiplication_properties(a, b, c):
    assert a * b == b * a
    assert (a * b) * c == a * (b * c)
    assert a * (b + c) == a * b + a * c
    assert a * 1 == a
    assert a * 0 == 0

@pytest.mark.parametrize("a, b", tests_inversion)
def test_qi_inversion(a, b):
    assert a.get_reciprocal() == b

@pytest.mark.parametrize("a, b, c", tests_division)
def test_qi_division(a, b, c):
    assert a / b == c

@pytest.mark.parametrize("a, b, _c", tests_addition)
def test_qi_properties_inversion_division(a, b, _c):
    assert a.get_reciprocal() * a == 1
    assert a * a.get_reciprocal() == 1
    assert (a.get_reciprocal()).get_reciprocal() == a

    with pytest.raises(MathDomainError):
        qi(0).get_reciprocal()

    assert a / a == 1
    assert a / b ==  a * b.get_reciprocal()

    with pytest.raises(MathDomainError):
        a / 0
        
@pytest.mark.parametrize("a, n, b", tests_power)
def test_qi_power(a, n, b):
    assert a**n == b

@pytest.mark.parametrize("a, b, _c", tests_addition)
def test_qi_power_properties(a, b, _c):
    assert a**0 == 1
    assert a**1 == a
    assert a**2 == a * a
    assert a**3 == a * a * a

    for n in range(5):
        assert (a * b)**n == a**n * b**n
        for m in range(5):
            assert a**m * a**n == a**(m + n)
            assert (a**m)**n == a**(m * n)

    with pytest.raises(MathDomainError):
        qi(0)**0

    with pytest.raises(MathDomainError):
        qi(0)**(-1)

    with pytest.raises(MathDomainError):
        qi(0)**(-2)

    with pytest.raises(MathDomainError):
        qi(0)**(-10)

@pytest.mark.parametrize("a, b, c", tests_addition)
def test_qi_general_identities(a, b, c):
    assert (a + b) - b == a
    assert (a - b) + b == a
    assert (a * b)/b == a
    assert (a / b) * b == a
    assert a * (b + c) - a * b - a * c == 0
    assert (a + b)**2 == a**2 + 2 * a * b + b**2
    assert (a - b)**2 == a**2 - 2 * a * b + b**2
    assert (a - b)*(a + b) == a**2 - b**2
    assert (a + b)**3 == a**3 + 3 * a**2 * b + 3 * a * b**2 + b**3
