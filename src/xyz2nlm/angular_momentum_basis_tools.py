from abc import ABC, abstractmethod
import numpy as np

class AngularMomentumBasis(ABC):
    def __init__(self, jmax):
        jmax_sanitized = self.sanitize_jmax(jmax)
        self.jmax = jmax_sanitized
        self.Nstates = self.jm2index(self.jmax, self.jmax) + 1
        self.index_list = self._jm_list()

    @abstractmethod
    def sanitize_jmax(self, jmax):
        pass

    @abstractmethod
    def jm2index(self, j, m):
        pass

    @abstractmethod
    def index2jm(self, j, m):
        pass

    def _jm_list(self):
        indices = np.arange(self.Nstates)
        jlist, mlist = self.index2jm(indices)
        return np.concatenate((jlist[:, None], mlist[:, None]), axis=1)


class FullAngularMomentumBasis(AngularMomentumBasis):
    def __init__(self, jmax):
        super().__init__(jmax)

    def sanitize_jmax(self, jmax):
        return np.floor(2*jmax).astype(int)/2.0

    def jm2index(self, j, m):
        return 2 *j*(j + 1) + m

    def index2jm(self, index):
        j = np.floor(0.5*(np.sqrt(8*index +1)-1)).astype(int)/2.0
        m = index - 2 * j * (j + 1)
        return j, m


class IntegerAngularMomentumBasis(AngularMomentumBasis):
    def __init__(self, jmax):
        super().__init__(jmax)

    def sanitize_jmax(self, jmax):
        return np.floor(jmax).astype(int)

    def jm2index(self, j, m):
        return j*(j + 1) + m

    def index2jm(self, index):
        j = np.floor(np.sqrt(index)).astype(int)
        m = index - j * (j + 1)
        return j, m


class EvenAngularMomentumBasis(AngularMomentumBasis):
    def __init__(self, jmax):
        super().__init__(jmax)

    def sanitize_jmax(self, jmax):
        jmax_int = np.floor(jmax).astype(int)
        return jmax_int if (jmax_int % 2 == 0) else jmax_int - 1

    def jm2index(self, j, m):
        return (j*(j + 1))//2 + m

    def index2jm(self, index):
        j = 2 * np.floor(0.25*(1 + np.sqrt(1+8*index))).astype(int)
        m = index - (j * (j + 1))//2
        return j, m


class OddAngularMomentumBasis(AngularMomentumBasis):
    def __init__(self, jmax):
        super().__init__(jmax)

    def sanitize_jmax(self, jmax):
        jmax_int = np.floor(jmax).astype(int)
        return jmax_int if (jmax_int % 2 == 1) else jmax_int - 1

    def jm2index(self, j, m):
        return (j*(j + 1))//2 + m

    def index2jm(self, index):
        j = 2 * np.floor(0.25*(-1 + np.sqrt(1+8*index))).astype(int) + 1
        m = index - (j * (j + 1))//2
        return j, m
