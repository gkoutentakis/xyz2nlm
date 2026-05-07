import numpy as np
from scipy.special import gammaln
import math

from mlx_create_c import inv_occ_mat, create_bos_ns
from scipy.sparse import coo_array
from ._angular_momentum_basis_tools import IntegerAngularMomentumBasis
from .custom_dataclasses import SphericalHOBasisState

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

def create_seed_state(N, L):

    if N <= 0 or L < 0 or (N + L)%2 != 0:
        raise ValueError("The parameters N={N} and L={L} should satisfy:"
                         " N>0, L >=0 and N,L same parity")

    q = (N - L)//2

    # create configurations
    confs = np.empty((0, 3), dtype=np.int64)
    for nz in range(0, N-L+1, 2):
        confs_2d = create_bos_ns(N - nz, 2)
        pad = nz * np.ones((confs_2d.shape[0], 1), dtype=confs_2d.dtype)
        confs_new = np.hstack((confs_2d, pad))
        confs = np.vstack((confs, confs_new))

    # integer work
    nx = confs[:, 0]
    nz = confs[:, 2]
    s = q - nz//2

    Sl_list = [Sls(L, ess) for ess in range(q+1)]
    list_S = [sign_and_log(Sl_list[s[i]][nx[i]]) for i in range(s.shape[0])]
    signs, Slogs = map(np.array, zip(*list_S))

    # remove zeros from arrays
    valid = signs != 0
    confs = confs[valid, :]
    Slogs = Slogs[valid]
    signs = signs[valid]

    nx = confs[:, 0]
    ny = confs[:, 1]
    nz = confs[:, 2]

    # coefficients
    coefficients = signs * coefficient_seed_state_expansion(N, L, nx, ny, nz, Slogs)

    # indices
    inds = inv_occ_mat(confs)

    return SphericalHOBasisState(n = N, l = L, m = L,
                                 non_zero_ind=inds,
                                 non_zero_val=coefficients)





