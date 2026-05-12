import numpy as np
from scipy.special import gammaln
import math

from mlx_create_c import inv_occ_mat, create_bos_ns
from .custom_dataclasses import SphericalHOBasisStateFactory
from .qi_interface import qi, qi_binom, qi_phase_i, qi_sqrt_fraction

def coefficient_seed_state_expansion(N, L, nx, ny, nz, Slog):

    q = (N - L)// 2
    s = q - nz//2

    log_positive = (
        Slog
        -0.5 * L * np.log(2.0)
        + (
            gammaln(q + 1)
            -gammaln(s + 1)
            -gammaln(q - s + 1)
          )
        + 0.5 * (
            gammaln(2 * L + 1 + 1)
            + gammaln(L + q + 1)
            + gammaln(nx + 1)
            + gammaln(ny + 1)
            + gammaln(nz + 1)
            - 2 * gammaln(L + 1)
            - gammaln(q + 1)
            - gammaln(N + L + 1 + 1)
        )
    )

    phase = 1j ** (3 * L - nx)
    out = phase * np.exp(log_positive)

    return out

def Sls(L, s):

    L = int(L)
    s = int(s)

    Slsn = [1, L]
    for n in range(2, L + 2 * s + 1):
        Slsnn = (L * Slsn[n-1] - (L + 2*s -n +2) * Slsn[n-2])//n
        Slsn.append(Slsnn)

    return Slsn
    
def sign_and_log(n):
    if n < 0:
        return -1, math.log(-n)

    if n==0:
        return 0, 0.0

    return +1, math.log(n)

def coefficient_seed_state_expansion_qi(N, L, nx, ny, nz, S):
    N = int(N)
    L = int(L)
    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
    S = int(S)

    if S == 0:
        return qi(0)

    q = (N - L) // 2
    s = q - nz // 2

    numerator = (
        math.factorial(2 * L + 1)
        * math.factorial(L + q)
        * math.factorial(nx)
        * math.factorial(ny)
        * math.factorial(nz)
    )
    denominator = (
        (2**L)
        * (math.factorial(L) ** 2)
        * math.factorial(q)
        * math.factorial(N + L + 1)
    )

    sign = -1 if S < 0 else 1
    return (
        qi(sign)
        * qi_phase_i(3 * L - nx)
        * qi(abs(S))
        * qi_binom(q, s)
        * qi_sqrt_fraction(numerator, denominator)
    )

def validate_seed_parameters(N, L):
    N = int(N)
    L = int(L)

    if N < 0 or L < 0 or N < L or (N + L)%2 != 0:
        raise ValueError("The parameters N={N} and L={L} should satisfy:"
                         " N>=0, N>=L, L >=0 and N,L same parity")
    return N, L


def create_seed_configurations(N, L):
    q = (N - L)//2

    confs = np.empty((0, 3), dtype=np.int64)
    for nz in range(0, N-L+1, 2):
        confs_2d = create_bos_ns(N - nz, 2)
        pad = nz * np.ones((confs_2d.shape[0], 1), dtype=confs_2d.dtype)
        confs_new = np.hstack((confs_2d, pad))
        confs = np.vstack((confs, confs_new))

    return confs, q


def evaluate_seed_integer_coefficients(L, q, confs):
    nx = confs[:, 0]
    nz = confs[:, 2]
    s = q - nz//2

    Sl_list = [Sls(L, ess) for ess in range(q + 1)]
    S_values = np.asarray([Sl_list[s[i]][nx[i]] for i in range(s.shape[0])], dtype=object)

    valid = S_values != 0
    return confs[valid, :], S_values[valid]


def build_coefficient_array_qi(N, L, confs, S_values):
    coefficients = [
        coefficient_seed_state_expansion_qi(N, L, int(nx), int(ny), int(nz), S)
        for (nx, ny, nz), S in zip(confs, S_values)
    ]
    return np.asarray(coefficients, dtype=object)


def build_coefficient_array_complex(N, L, confs, S_values):
    if len(S_values) == 0:
        return np.asarray([], dtype=complex)

    nx = confs[:, 0]
    ny = confs[:, 1]
    nz = confs[:, 2]
    list_S = [sign_and_log(S) for S in S_values]
    signs, Slogs = map(np.array, zip(*list_S))
    return signs * coefficient_seed_state_expansion(N, L, nx, ny, nz, Slogs)


def create_seed_state(N, L, exact=True):
    N, L = validate_seed_parameters(N, L)
    confs, q = create_seed_configurations(N, L)
    confs, S_values = evaluate_seed_integer_coefficients(L, q, confs)

    if exact:
        coefficients = build_coefficient_array_qi(N, L, confs, S_values)
    else:
        coefficients = build_coefficient_array_complex(N, L, confs, S_values)

    return SphericalHOBasisStateFactory(n=N, l=L, m=L,
                                        non_zero_ind=inv_occ_mat(confs),
                                        non_zero_val=coefficients)
