import math
import sympy as sp
import sympy.ntheory as nt
import gmpy2

from .qi_algebra import QINumber

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
