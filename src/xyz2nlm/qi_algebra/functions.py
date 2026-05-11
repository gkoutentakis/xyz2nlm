import sympy as sp
import sympy.ntheory as nt

from .number import QINumber

def factorial_prime_factors(n):
    return nt.factorint(sp.factorial(n))

def integer_prime_factors(n):
    return nt.factorint(n)

def multiply_prime_factors(fact_dict_running_sum, fact_dict_to_sum):
    for key in fact_dict_to_sum.keys():
        if key in fact_dict_running_sum.keys():
            fact_dict_running_sum[key] += fact_dict_to_sum[key]
        else:
            fact_dict_running_sum[key] = fact_dict_to_sum[key]
    return fact_dict_running_sum

def multiply_many_prime_factors(list_of_dicts):
    if len(list_of_dicts) == 0:
        return {}

    dict_running_sum = list_of_dicts.pop()
    for dict_to_sum in list_of_dicts:
        dict_running_sum = multiply_prime_factors(dict_running_sum, dict_to_sum)
    return dict_running_sum

def factorization_to_qi(prime_factors):
    integer = int(1)
    number_of_sqrt = int(0)
    for prime, exponent in prime_factors.items():
        if exponent%2 == 0:
            integer *= prime**(exponent//2)
        elif exponent%2 == 1:
            integer *= prime**((exponent-1)//2)
            number_of_sqrt += 2**(sp.primepi(prime)-1)

    result = QINumber()
    result.pack(integer, 0, 1, number_of_sqrt)
    return result

def square_root_factorial(n):
    prime_factors = factorial_prime_factors(n)
    return factorization_to_qi(prime_factors)

def square_root_integer(n):
    return QINumber(n=int(n), sqrt=True)

def square_root_fraction(numerator, denominator):
    numerator = int(numerator)
    denominator = int(denominator)
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator < 0:
        raise ValueError("QINumber real square roots require a non-negative numerator")
    if numerator == 0:
        return QINumber(0)
    return square_root_integer(numerator) / square_root_integer(denominator)

def conjugate(number):
    number = QINumber.convert_if_needed(number)
    result = QINumber(0)
    for sqrt_order in number.get_square_root_indices():
        real, imag, denominator = number.unpack_sqrt_values(sqrt_order)
        term = QINumber(0)
        term.pack(real, -imag, denominator, sqrt_order)
        result += term
    return result

def is_zero(number):
    return QINumber.convert_if_needed(number) == QINumber(0)

def to_complex(number):
    number = QINumber.convert_if_needed(number)
    out = 0j
    for sqrt_order in number.get_square_root_indices():
        real, imag, denominator = number.unpack_sqrt_values(sqrt_order)
        radicand = QINumber.order_to_radicand(sqrt_order)
        out += complex(real, imag) / denominator * (radicand ** 0.5)
    return out
