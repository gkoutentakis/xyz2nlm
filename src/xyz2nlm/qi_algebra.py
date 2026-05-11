import math
import sympy as sp
import sympy.ntheory as nt
import gmpy2

import copy

from .qi_utils import ascii_radical_sum

class MathDomainError(ArithmeticError):
    """Raised when an operation is not defined within the mathematical domain of
 Q(i, sqrt(r_1), ...) numbers."""

class QINumber:
    """QINumber implments the exact arithmetic of numbers corresponding to
finite sums with terms corresponding to the Gaussian integers times a radical.
A = sum_{j = 1}^N_{terms} (p_j + i*q_j) / d_j  sqrt{r_j}

The radicands are encoded by bitmasks where the bit k records whether the k-th
 prime occurs with odd exponent in the expansion of the radicand in prime factors.

The underlying data maps this squareclass bitmask -> Gaussian rational coefficient
[p, q, d]. [p, q, d] means (p + i*q) / d, with d > 0 and gcd(p, q, d) = 1.
"""
    def __init__(self, n=0, sqrt=False, copy=None):
        if copy is not None:
            self.copy(copy)
            return

        if n==0:
            self.set_zero()
            return

        if n==1:
            self.set_one()
            return

        if n==1j and not sqrt:
            self.set_imaginary_unity()
            return

        if isinstance(n, int) and not sqrt:
            self.pack(n, 0, 1, 0)
            return

        if isinstance(n, int) and sqrt:
            self.pack_root_int(n)
            return

        if isinstance(n, float) and not sqrt:
            num, den = n.as_integer_ratio()
            self.pack(num, 0, den, 0)
            return

        if isinstance(n, float) and sqrt:
            num, den = math.sqrt(n).as_integer_ratio()
            self.pack(num, 0, den, 0)
            return

        if isinstance(n, complex) and not sqrt:
            self.pack_complex(n)
            return

        if isinstance(n, complex) and sqrt:
            if n.imag != 0:
                raise MathDomainError("The square root of a complex number is ambiguous"
                                      "due to the branch cut. Please calculate the square root"
                                      " in the required branch outside the QINumber package"
                                      " and initialize with the result and sqrt=False.")

            n = n**0.5
            self.pack_complex(n)
            return

    # calculations with QI numbers
    def __add__(self, other):
        other = self.convert_if_needed(other)
        return self.sum_qi_unsafe(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = self.convert_if_needed(other)
        other = other.get_minus()
        return self.sum_qi_unsafe(self, other)
    
    def __rsub__(self, other):
        other = self.convert_if_needed(other)
        return other - self

    def __neg__(self):
        return self.get_minus()

    def __mul__(self, other):
        other = self.convert_if_needed(other)
        return self.prod_qi_unsafe(self, other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = self.convert_if_needed(other)
        other = other.get_reciprocal()
        return self.prod_qi_unsafe(self, other)

    def __pow__(self, n):
        assert isinstance(n, int)
        return self.power_qi(n)

    def __eq__(self, other):
        other = self.convert_if_needed(other)
        return self.check_equal(self, other)

    def __str__(self):
        return self.print()

    def __repr__(self):
        return f"{type(self).__name__}\n{self!s}" 

    # special number shortcuts
    def set_zero(self):
        self._storage = {}

    def set_one(self):
        self._storage = {0: [1, 0, 1]}

    def set_imaginary_unity(self):
        self._storage = {0: [0, 1, 1]}

    # packing routines
    def copy(self, other):
        if isinstance(other, type(self)):
            self._storage = copy.deepcopy(other._storage)

    def pack(self, numerator_real, numerator_imag, denominator, sqrt_order):
        numerator_real, numerator_imag, denominator = \
            self.normalize_rational_part(numerator_real, numerator_imag, denominator)
        self._storage = {sqrt_order:[numerator_real, numerator_imag, denominator]}

    def pack_complex(self, n):
        nr = n.real
        ni = n.imag
        num_r, den_r = nr.as_integer_ratio()
        num_i, den_i = ni.as_integer_ratio()
        num_r *= den_i
        num_i *= den_r
        den = den_r * den_i
        self.pack(num_r, num_i, den, 0)

    def pack_root_int(self, n):
        prime_factors = nt.factorint(n)
        sqrt_num, ints = self._factorized_root_to_qi(prime_factors)
        self._storage = {sqrt_num: ints}

    # methods
    @classmethod
    def require_qi_for_op(cls, name, *args):
        errors = []
        for i, arg in enumerate(args):
            if not isinstance(arg, type(cls)):
                errors.append((i, arg, "type"))
                continue
            if not arg._is_valid():
                errors.append((i, arg, "invalid"))
        if errors:
            error_string = f"{name} requires all inputs to be QINumbers:" + \
                f"{len(errors)} occured:" + \
                (f"argument {i} is of type {type(arg)}" if reason=="type" else \
                 f"argument {i} is invalid QINumber"
                 for i, arg, reason in zip(errors))
            raise TypeError(error_string)
        
        return self.sum_qi_unsafe(a, b)

    @classmethod
    def sum_qi(cls, a, b):
        cls.require_qi_for_op("sum_qi", a, b)
        return self.sum_qi_unsafe(a, b)

    @classmethod
    def sum_qi_unsafe(cls, a, b):
        result = cls(copy=b)

        for sqrt, value in a._storage.items():
            if sqrt not in b._storage.keys():
                result._storage[sqrt] = value
            else:
                ar, ai, ad = a.unpack_sqrt_values(sqrt)
                br, bi, bd = b.unpack_sqrt_values(sqrt)
                real_new = ar * bd + br * ad
                imag_new = ai * bd + bi * ad
                denominator_new = bd * ad
                real_new, imag_new, denominator_new = \
                    cls.normalize_rational_part(real_new, imag_new, denominator_new)
                result._storage[sqrt] = [real_new, imag_new, denominator_new]

        result.eliminate_zero_terms()
        return result

    @classmethod
    def prod_qi(cls, a, b):
        cls.require_qi_for_op("prod_qi", a, b)
        return self.prod_qi_unsafe(a, b)

    @classmethod
    def prod_qi_unsafe(cls, a, b):
        result = cls(n=0)
        for sqrt_a in a._storage.keys():
            for sqrt_b in b._storage.keys():
                sqrt_new = sqrt_a ^ sqrt_b
                int_multiplier = cls.order_to_radicand(sqrt_a & sqrt_b)

                ar, ai, ad = a.unpack_sqrt_values(sqrt_a)
                br, bi, bd = b.unpack_sqrt_values(sqrt_b)

                real_new = int_multiplier * (ar * br - ai * bi)
                imag_new = int_multiplier * (ai * br + ar * bi)
                denominator_new = ad * bd

                real_new, imag_new, denominator_new = \
                    cls.normalize_rational_part(real_new, imag_new, denominator_new)
                term = cls()
                term.pack(real_new, imag_new, denominator_new, sqrt_new)
                result += term
        result.eliminate_zero_terms()
        return result

    @classmethod
    def order_to_radicand(cls, order):
        radicand = 1
        list_of_binary_exponents = list(gmpy2.xmpz(order).iter_set())
        for exponent in list_of_binary_exponents:
            radicand *= nt.prime(exponent+1)
        return radicand

    @classmethod
    def normalize_rational_part(cls, numerator_real, numerator_imag, denominator):
        if denominator == 0:
            raise ValueError("denominator of zero detected")
        if denominator < 0:
            numerator_real = -numerator_real
            numerator_imag = -numerator_imag
            denominator = -denominator

        normalization = math.gcd(numerator_real, numerator_imag, denominator)

        numerator_real = numerator_real//normalization
        numerator_imag = numerator_imag//normalization
        denominator = denominator//normalization

        return numerator_real, numerator_imag, denominator

    @classmethod
    def check_equal(cls, a, b):
        return a._storage == b._storage

    @classmethod
    def convert_if_needed(cls, value):
        if isinstance(value, cls):
            return value

        if isinstance(value, int) or \
           isinstance(value, float) or \
            isinstance(value, complex):
            return cls(n=value, sqrt=False)

        raise TypeError(f"{value} cannot be converted to QINumber")

    # instance methods
    def get_rank_of_indices(self, sqrt_inds):
        sqrt_inds = sqrt_inds.copy()
        rank = 0
        while sqrt_inds:
            index = max(sqrt_inds)

            if index == 0:
                break

            sqrt_inds.remove(index)
            rank += 1

            sqrt_inds = [sqrt_ind ^ 1 << (int(index).bit_length() - 1) for sqrt_ind in sqrt_inds]
        return rank

    def get_rank_and_pivots(self):
        sqrt_inds = self.get_square_root_indices()
        sqrt_inds.sort(reverse=True)
        basis = []
        for index in sqrt_inds:

            for vector in basis:
                index = min(index, vector ^ index)

            if index > 0:
                for i, vector in enumerate(basis):
                    basis[i] = min(vector, vector ^ index)

                basis.append(index)

        # basis.sort(reverse=True)
        rank = len(basis)

        return rank, basis

    def get_support(self, pivots):
        sqrt_inds = self.get_square_root_indices()
        supports = {}
        for pivot in pivots:
            pivot_bit = 1 << (int(pivot).bit_length() - 1)
            supports[pivot] = [sqrt_ind for sqrt_ind in sqrt_inds \
                               if pivot_bit & sqrt_ind > 0]
        return supports


    def set_terms_to_minus(self, sqrt_inds):
        result = type(self)(copy=self)
        for sqrts in sqrt_inds:
            result._storage[sqrts][0] *= -1
            result._storage[sqrts][1] *= -1
        return result

    def get_minus(self):
        result = type(self)(copy=self)
        for sqrt, values in result._storage.items():
            values[0] *= -1
            values[1] *= -1
        return result

    def get_reciprocal(self):
        number_of_terms = self.number_of_terms()

        if number_of_terms == 0:
            raise MathDomainError("Division by zero is undefined.")

        if number_of_terms == 1:
            return self.get_reciprocal_single_term()

        if number_of_terms == 2:
            return self.get_reciprocal_two_terms()

        if number_of_terms == 3:
            return self.get_reciprocal_three_terms()

        return self.get_reciprocal_greedy()

    def get_reciprocal_single_term(self):
        sqrt_ind = self.get_square_root_indices()[0]
        r, i, d = self.unpack_sqrt_values(sqrt_ind)
        r_n = r * d
        i_n = -i * d
        d_n = (r**2 + i**2) * self.order_to_radicand(sqrt_ind)

        r_n, i_n, d_n = self.normalize_rational_part(r_n, i_n, d_n)
        result = type(self)()
        result.pack(r_n, i_n, d_n, sqrt_ind)
        return result

    def get_reciprocal_two_terms(self):
        sqrt_inds = self.get_square_root_indices()
        N = type(self)(n=1)
        D = type(self)(copy=self)

        factor = type(self)(copy=D).set_terms_to_minus([sqrt_inds[0]])

        N *= factor
        D *= factor

        return N * D.get_reciprocal_single_term()

    def get_reciprocal_three_terms(self):
        sqrt_inds = self.get_square_root_indices()

        rank, pivots = self.get_rank_and_pivots()
        support_dict = self.get_support(pivots)
        for pivot, supports in support_dict.items():
            if len(supports) == 1:
                sqrt_inds.remove(supports[0])
                sqrt_inds.append(supports[0])
                break

        N = type(self)(n=1)
        D = type(self)(copy=self)

        factor = type(self)(copy=D).set_terms_to_minus([sqrt_inds[-1]])

        N *= factor
        D *= factor

        if D.number_of_terms == 1:
            return N * D.get_reciprocal_single_term()

        sqrt_inds = D.get_square_root_indices()
        factor = type(self)(copy=D).set_terms_to_minus([sqrt_inds[-1]])

        N *= factor
        D *= factor

        return N * D.get_reciprocal_single_term()

    def get_reciprocal_greedy(self):
        N = type(self)(n=1)
        D = type(self)(copy=self)

        while D.has_radical():
            sqrt_inds = D.get_square_root_indices()
            rank, pivots = D.get_rank_and_pivots()

            if rank > 1:
                support_dict = D.get_support(pivots)

                terms_pivot = {}
                for pivot, supports in support_dict.items():
                    terms_supports = D.get_rank_of_indices(supports)
                    terms_non_supports = D.get_rank_of_indices([index for index in sqrt_inds if index not in supports])
                    terms_pivot[pivot] = terms_supports + terms_non_supports

                optimal_pivot = min(terms_pivot, key=terms_pivot.get)

                factor = type(self)(copy=D).set_terms_to_minus(support_dict[optimal_pivot])
            else:
                factor = type(self)(copy=D).set_terms_to_minus([max(sqrt_inds)])

            N *= factor
            D *= factor

        return N * D.get_reciprocal_single_term()

    def power_qi(self, n):
        if n < 0:
            inverse = self.get_reciprocal()
            return inverse**(-n)

        if n == 0 and self == 0:
            raise MathDomainError("Exponentiation 0^0 is undefined.")

        if n == 0:
            return type(self)(n=1)

        if n == 1:
            return type(self)(copy=self)

        # too easy to bother category
        if n == 2:
            return self * self

        if self.number_of_terms() == 1:
            return self.power_qi_single_term(n)

        # if everything fails just multiply
        return math.prod([self for i in range(n)])

    def power_qi_single_term(self, n):
        square_root_index = self.get_square_root_indices()[0]
        square_numerator_real, square_numerator_imag, square_denominator = \
            self.get_square(square_root_index).unpack_sqrt_values(0)

        if n%2 == 0:
            exponent = n//2
            result = type(self)(n=1)

        if n%2 == 1:
            exponent = (n-1)//2
            result = type(self)(copy=self)

        # consider making a class for Gaussian integers and not do these things
        # inside QINumber as it is not intended for doing exponentiation for
        # Gaussian integers that is what the problem results to here

        factor = type(self)()
        if square_numerator_imag == 0:
            factor.pack(square_numerator_real**exponent, 0, square_denominator**exponent, 0)
            return result * factor

        if square_numerator_real == 0:
            if exponent%2 == 0:
                factor.pack((-1)**(exponent//2) * square_numerator_imag**exponent,
                            0,
                            square_denominator**exponent,
                            0)
                return result * factor
            else:
                factor.pack(0,
                            (-1)**((exponent-1)//2) * square_numerator_imag**exponent,
                            square_denominator**exponent,
                            0)
                return result * factor

        # express complex number in tuples and exponentiate by squaring
        multiply_tuples = lambda tuple1, tuple2: \
            (tuple1[0] * tuple2[0] - tuple1[1] * tuple2[1], \
             tuple1[0] * tuple2[1] + tuple1[1] * tuple2[0])

        result_tuple = (1, 0)
        base_tuple = (square_numerator_real, square_numerator_imag)
        while exponent>0:
            if exponent & 1:
                result_tuple = multiply_tuples(result_tuple, base_tuple)
            base_tuple = multiply_tuples(base_tuple, base_tuple)
            exponent >>= 1

        factor.pack(result_tuple[0], result_tuple[1], square_denominator**exponent, 0)
        return result * factor

    def get_real(self):
        result = type(self)(copy=self)
        for sqrt, values in result._storage.keys():
            values[1] *= 0
        result.eliminate_zero_terms()
        return result

    def get_imag(self):
        result = type(self)(copy=self)
        for sqrt, values in result._storage.keys():
            values[0] *= 0
        result.eliminate_zero_terms()
        return result

    def get_square(self, sqrt_ind):
        r, i, d = self.unpack_sqrt_values(sqrt_ind)
        radicand = self.order_to_radicand(sqrt_ind)
        r_n = (r**2 - i**2) * radicand 
        i_n = 2 * r * i * radicand 
        d_n = d**2

        r_n, i_n, d_n = self.normalize_rational_part(r_n, i_n, d_n)
        result = type(self)()
        result.pack(r_n, i_n, d_n, 0)
        return result

    def get_magnitude_squared(self, sqrt_ind):
        r, i, d = self.unpack_sqrt_values(sqrt_ind)
        r_n = (r**2 + i**2) * self.order_to_radicand(sqrt_ind)
        i_n = 0
        d_n = d**2

        r_n, i_n, d_n = self.normalize_rational_part(r_n, i_n, d_n)
        result = type(self)()
        result.pack(r_n, i_n, d_n, 0)
        return result

    def _is_valid(self):
        if not isinstance(self._storage, dict):
            return False

        for sqrt, value in self._storage.keys():
            if value[2] <= 0:
                return False

        return True

    def has_radical(self):
        if self == type(self)(n=0):
            return False

        if max(self.get_square_root_indices()) > 0:
            return True

        return False

    def _factorized_root_to_qi(self, prime_factors):
        integer = int(1)
        number_of_sqrt = int(0)
        for prime, exponent in prime_factors.items():
            if exponent%2 == 0:
                integer *= prime**(exponent//2)
            elif exponent%2 == 1:
                integer *= prime**((exponent-1)//2)
                number_of_sqrt += 2**(sp.primepi(prime)-1)
        return number_of_sqrt, [integer, 0, 1]

    # normalization
    def eliminate_zero_terms(self):
        keys_to_pop = []
        for sqrt_ind in self._storage.keys():
            r, i, d = self.unpack_sqrt_values(sqrt_ind)
            if r == 0 and i == 0:
                keys_to_pop.append(sqrt_ind)

        for key in keys_to_pop:
            self._storage.pop(key)

    # IO methods
    def unpack_sqrt_values(self, sqrt_order):
        list_to_unpack = self._storage[sqrt_order]
        numerator_real = list_to_unpack[0]
        numerator_imag = list_to_unpack[1]
        denominator = list_to_unpack[2]
        return numerator_real, numerator_imag, denominator

    def get_square_root_indices(self):
        return list(self._storage.keys())
        
    def number_of_terms(self):
        return len(self._storage)

    def print(self, mode="fancy"):
        # serialize
        terms = []
        for sqrt_ind in self._storage.keys():
            num_r, num_i, den = self.unpack_sqrt_values(sqrt_ind)
            radicand = self.order_to_radicand(sqrt_ind)
            terms.append((num_r, num_i, den, radicand))

        return ascii_radical_sum(terms)

