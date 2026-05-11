import concurrent.futures as cf
from os import cpu_count

import mlx_create_c
import numpy as np
import numpy.testing as npt

from ._angular_momentum_basis_tools import EvenAngularMomentumBasis, OddAngularMomentumBasis
from .create_angular_momentum_matrices import CreateAngularMomentumMatrices
from .custom_dataclasses import (
    SphericalHOBasisStateNP,
    SphericalHOBasisStateQI,
    TransientStateNP,
    TransientStateQI,
)
from .direct_seed_state_calculation import create_seed_state
from .qi_interface import (
    exact_zero_array,
    qi,
    qi_i,
    qi_sqrt_fraction,
    qi_sqrt_int,
    to_complex_array,
)


class BlockLmApplication:
    def __init__(self, n, block_in, ls, ms, exact=True):
        self.exact = bool(exact)
        self.am = CreateAngularMomentumMatrices(n, exact=exact)
        _, self.Lm = self.am.get_ladder_operators(operator_buffer_size=self.am.basis_size)

        self.work_array_a = block_in
        self.work_array_b = self.am._Lm_buffer

        self.states_current = self.work_array_a
        self.states_current_using_buffer_a = True

        self.block_ls = ls.copy()
        self.block_ms = ms.copy()
        self._configure_implementation()

    def _configure_implementation(self):
        if self.exact:
            self._normalize_states = self._normalize_states_qi
        else:
            self._normalize_states = self._normalize_states_complex

    def decrease_m_of_block(self):
        self.states_current = self.Lm @ self.states_current
        self.update_working_buffer_data()
        self.normalize_states_after_lm_application()

        self.block_ms -= 1

        return self.states_current, self.block_ls, self.block_ms

    def update_working_buffer_data(self):
        if self.states_current_using_buffer_a:
            self.am._change_buffer_of_operator("Lm", self.work_array_a)
            self.states_current_using_buffer_a = False
        else:
            self.am._change_buffer_of_operator("Lm", self.work_array_b)
            self.states_current_using_buffer_a = True

    def normalize_states_after_lm_application(self):
        self._normalize_states()

    def _normalize_states_complex(self):
        self.states_current /= np.sqrt(
            (self.block_ls + self.block_ms) * (self.block_ls - self.block_ms + 1)
        )[None, :]

    def _normalize_states_qi(self):
        for column, (ell, emm) in enumerate(zip(self.block_ls, self.block_ms)):
            normalization = qi_sqrt_int((ell + emm) * (ell - emm + 1))
            self.states_current[:, column] = [
                value / normalization for value in self.states_current[:, column]
            ]

    def drop_first_state_from_block(self):
        self.states_current = self.states_current[:, 1:]
        self.block_ls = self.block_ls[1:]
        self.block_ms = self.block_ms[1:]


class SphericalHOBasis:
    """Construct spherical harmonic-oscillator states from Cartesian states.

    Exact mode is the default. In exact mode coefficients are QINumber algebraic
    numbers and dense shell blocks use object dtype. Use as_complex_ndarray()
    helpers or as_complex_basis_states() for explicit numerical inspection.
    """

    def __init__(self, Nmax, exact=True):
        self.Nmax = Nmax
        self.exact = bool(exact)
        self._configure_implementation()
        self._define_lambdas()
        self._set_basis_state_matrices()
        self.initialize_calculation()

    def _configure_implementation(self):
        if self.exact:
            self.state_class = SphericalHOBasisStateQI
            self.transient_state_class = TransientStateQI
            self._one = qi(1)
            self._minus_one = qi(-1)
            self._two_i = qi(2) * qi_i()
            self._create_seed_state = self._create_seed_state_qi
            self._initial_even_values = self._initial_even_values_qi
            self._initial_odd_values = self._initial_odd_values_qi
            self._new_basis_storage = self._new_basis_storage_qi
            self._basis_states_to_complex = self._basis_states_to_complex_qi
            self._basis_state_views_to_complex = self._basis_state_views_to_complex_qi
            self._s2_weight = self._s2_weight_qi
            self._r22_parameters = self._r22_parameters_qi
            self._adag_impl = self._adag_qi
            return

        self.state_class = SphericalHOBasisStateNP
        self.transient_state_class = TransientStateNP
        self._one = 1.0
        self._minus_one = -1
        self._two_i = 2j
        self._create_seed_state = self._create_seed_state_complex
        self._initial_even_values = self._initial_even_values_complex
        self._initial_odd_values = self._initial_odd_values_complex
        self._new_basis_storage = self._new_basis_storage_complex
        self._basis_states_to_complex = self._basis_states_to_complex_complex
        self._basis_state_views_to_complex = self._basis_state_views_to_complex_complex
        self._s2_weight = self._s2_weight_complex
        self._r22_parameters = self._r22_parameters_complex
        self._adag_impl = self._adag_complex

    def _initial_state_even_sector(self):
        return self.state_class(
            n=0,
            l=0,
            m=0,
            non_zero_ind=np.array([0]),
            non_zero_val=self._initial_even_values(),
        )

    def _initial_even_values_qi(self):
        return np.asarray([self._one], dtype=object)

    def _initial_even_values_complex(self):
        return np.asarray([self._one])

    def _initial_state_odd_sector(self):
        return self.state_class(
            n=1,
            l=1,
            m=1,
            non_zero_ind=np.array([0, 1]),
            non_zero_val=self._initial_odd_values(),
        )

    def _initial_odd_values_qi(self):
        root_two = qi_sqrt_int(2)
        return np.asarray([-self._one / root_two, -qi_i() / root_two], dtype=object)

    def _initial_odd_values_complex(self):
        return np.array([-1.0 / np.sqrt(2), -1j / np.sqrt(2)])

    def initialize_calculation(self):
        self.seed_states = {}

        n = np.arange(self.Nmax + 1)[:, None]
        m = np.arange(self.Nmax + 1)[None, :]

        self.pending_seed_state_checklist = n >= m
        self.pending_seed_state_checklist *= (n + m) % 2 == 0

        self.available_seed_state_checklist = np.zeros_like(self.pending_seed_state_checklist)

        self.record_seed_state(self._initial_state_even_sector())
        if self.Nmax >= 1:
            self.record_seed_state(self._initial_state_odd_sector())

        self.pending_blocks_checklist = np.ones((self.Nmax + 1,), dtype=bool)
        self.running_blocks_checklist = np.zeros_like(self.pending_blocks_checklist)
        self.available_blocks_checklist = np.zeros_like(self.pending_blocks_checklist)
        self.update_block_availability()

        self.jobs_dispatched = {}

    def calculate_states(self, method="Threads", max_workers=None):
        if method == "Threads":
            executor = cf.ThreadPoolExecutor
        elif method == "Serial":
            executor = cf.ThreadPoolExecutor
            max_workers = 1
        else:
            raise ValueError(f"Unknown method: {method}\nUse either 'Threads', 'Serial'")

        proposed_workers = cpu_count() or 1
        self.workers = proposed_workers if (max_workers is None) \
            or (proposed_workers <= max_workers) else max_workers

        self.pool = executor(max_workers=self.workers)

        while not self.calculation_done():
            while len(self.jobs_dispatched) < self.workers:
                if not self.dispatch_job():
                    break

            done, _ = cf.wait(self.jobs_dispatched.keys(), return_when=cf.FIRST_COMPLETED)

            for future in done:
                self.process_result(future)
            self.update_block_availability()

    def process_result(self, future):
        job_type = self.jobs_dispatched.pop(future)

        if job_type == "Block":
            self.record_block(future.result())
            return

        if job_type == "Seed State":
            self.record_seed_state(future.result())
            return

        raise ValueError(f"Unknown job type {job_type}")

    def dispatch_job(self):
        if (n := self.next_available_block()) is not None:
            self.available_blocks_checklist[n] = False
            self.running_blocks_checklist[n] = True
            future = self.pool.submit(self.job_make_block, n)
            self.jobs_dispatched[future] = "Block"
            return True

        if (shape := self.next_available_seed_state()) is not None:
            n, l = shape
            self.available_seed_state_checklist[n, l] = False
            future = self.pool.submit(self.job_make_seed_state, n, l)
            self.jobs_dispatched[future] = "Seed State"
            return True

        return False

    def job_make_block(self, n):
        seed_states = [self.seed_states[(n, m)] for m in range(0 if n % 2 == 0 else 1, n + 1, 2)]
        self.complete_fixed_n_block_from_seed_states(seed_states)
        return n

    def record_block(self, n):
        self.pending_blocks_checklist[n] = False

    def job_make_seed_state(self, n, l):
        return self._create_seed_state(n, l)

    def _create_seed_state_qi(self, n, l):
        return create_seed_state(n, l, exact=True)

    def _create_seed_state_complex(self, n, l):
        return create_seed_state(n, l, exact=False)

    def record_seed_state(self, seed_state):
        n, l, m = seed_state.get_indices()

        self.seed_states[(n, m)] = seed_state
        self.pending_seed_state_checklist[n, m] = False

        if n + 2 > self.Nmax:
            return

        if l == n:
            self.available_seed_state_checklist[n + 2, l + 2] = True

        self.available_seed_state_checklist[n + 2, l] = True

    def next_available_seed_state(self):
        ind = np.flatnonzero(self.available_seed_state_checklist)

        if ind.shape[0] > 0:
            flat_ind = ind[0]
            return np.unravel_index(flat_ind, self.available_seed_state_checklist.shape)
        return None

    def next_available_block(self):
        ind = np.flatnonzero(self.available_blocks_checklist)

        if ind.shape[0] > 0:
            return ind[0]
        return None

    def update_block_availability(self):
        completed_seed_state_blocks = np.sum(self.pending_seed_state_checklist, axis=1) == 0
        self.available_blocks_checklist[completed_seed_state_blocks] = True
        self.available_blocks_checklist *= self.pending_blocks_checklist
        self.available_blocks_checklist *= np.logical_not(self.running_blocks_checklist)

    def calculation_done(self):
        return np.sum(self.pending_blocks_checklist) == 0

    def pack_list_spherical_basis_states(self, states):
        indices = np.array([list(state.get_indices()) for state in states])

        n = indices[0, 0]
        expected_seed_count = n // 2 + 1 if n % 2 == 0 else (n + 1) // 2
        assert indices.shape[0] == expected_seed_count
        npt.assert_array_equal(indices[:, 0], n)
        npt.assert_array_equal(indices[:, 1], indices[:, 2])

        order = np.argsort(indices[:, 1])
        assert indices[order[-1], 1] == n

        out = np.column_stack([state.as_ndarray() for state in states])
        return n, out[:, order]

    def _set_basis_state_matrices(self):
        elements_sector_n = lambda n: ((n + 1) * (n + 2)) // 2
        number_of_elements_to_n = lambda n: (
            ((n + 1) * (n + 2) * (n + 3) * (3 * n**2 + 12 * n + 10)) // 60
            if n >= 0
            else 0
        )

        self.basis_states = self._new_basis_storage(number_of_elements_to_n(self.Nmax))

        self.basis_states_views_n = [
            self.basis_states[number_of_elements_to_n(i - 1) : number_of_elements_to_n(i)].reshape(
                (elements_sector_n(i), elements_sector_n(i)), copy=False
            )
            for i in range(self.Nmax + 1)
        ]

    def as_complex_basis_states(self):
        return self._basis_states_to_complex()

    def as_complex_basis_states_views_n(self):
        return self._basis_state_views_to_complex()

    def _new_basis_storage_qi(self, size):
        return exact_zero_array((size,))

    def _new_basis_storage_complex(self, size):
        return np.zeros((size,), dtype=complex)

    def _basis_states_to_complex_qi(self):
        return to_complex_array(self.basis_states)

    def _basis_states_to_complex_complex(self):
        return self.basis_states

    def _basis_state_views_to_complex_qi(self):
        return [to_complex_array(view) for view in self.basis_states_views_n]

    def _basis_state_views_to_complex_complex(self):
        return self.basis_states_views_n

    def s2_operator(self, state_in):
        n, l, m = state_in.get_indices()

        n_out = n + 2
        l_out = l
        m_out = m

        termx = self._axdag_sq(state_in)
        termy = self._aydag_sq(state_in)
        termz = self._azdag_sq(state_in)

        total_state = (termx + termy + termz).scale(self._s2_weight(n, l))

        return self.state_class(
            n=n_out,
            l=l_out,
            m=m_out,
            non_zero_ind=total_state.non_zero_ind,
            non_zero_val=total_state.non_zero_val,
        )

    def _s2_weight_qi(self, n, l):
        return self._one / qi_sqrt_int((n - l + 2) * (n + l + 3))

    def _s2_weight_complex(self, n, l):
        return ((n - l + 2) * (n + l + 3)) ** (-0.5)

    def r22_operator(self, state_in):
        n, l, m = state_in.get_indices()
        n_out = n + 2
        l_out = l + 2
        m_out = m + 2

        overall_weight, minus_one, two_i = self._r22_parameters(n, l)

        termx = self._axdag_sq(state_in)
        termy = self._aydag_sq(state_in).scale(minus_one)
        termxy = self._axdag(self._aydag(state_in)).scale(two_i)

        total_state = (termx + termy + termxy).scale(overall_weight)

        return self.state_class(
            n=n_out,
            l=l_out,
            m=m_out,
            non_zero_ind=total_state.non_zero_ind,
            non_zero_val=total_state.non_zero_val,
        )

    def _r22_parameters_qi(self, n, l):
        numerator = (2 * l + 3) * (2 * l + 5)
        denominator = 4 * (l + 1) * (l + 2) * (n + l + 3) * (n + l + 5)
        return qi_sqrt_fraction(numerator, denominator), self._minus_one, self._two_i

    def _r22_parameters_complex(self, n, l):
        overall_weight = 0.5 * np.sqrt(
            (2 * l + 3) * (2 * l + 5) / ((l + 1) * (l + 2) * (n + l + 3) * (n + l + 5))
        )
        return overall_weight, self._minus_one, self._two_i

    def complete_fixed_n_block_from_seed_states(self, seed_states):
        n, block_in = self.pack_list_spherical_basis_states(seed_states)
        if n == 0:
            self.basis_states_views_n[0][0] = self._one
            return

        block_out_view = self.basis_states_views_n[n]

        if n % 2 == 0:
            ind = EvenAngularMomentumBasis(n)
            block_out_view[:, ind.jm2index(0, 0)] = block_in[:, 0]
            block_in = block_in[:, 1:]
            ls = np.arange(2, n + 1, 2)
        else:
            ind = OddAngularMomentumBasis(n)
            ls = np.arange(1, n + 1, 2)

        block_out_view[:, ind.jm2index(ls, ls)] = block_in
        view_pos = id(block_out_view)

        block_controller = BlockLmApplication(n, block_in, ls, ls, exact=self.exact)
        for first_l_in_block in ls:
            for _ in range(4 if first_l_in_block != 1 else 2):
                states_current, ls_c, ms_c = block_controller.decrease_m_of_block()
                block_out_view[:, ind.jm2index(ls_c, ms_c)] = states_current
                assert id(block_out_view) == view_pos

            block_controller.drop_first_state_from_block()

    def _adag(self, state, ind):
        return self._adag_impl(state, ind)

    def _adag_complex(self, state, ind):
        indices, weight = mlx_create_c.creation_operator(state.non_zero_ind, ind, state.n, 3)
        return self.transient_state_class(
            n=state.n + 1,
            non_zero_ind=indices,
            non_zero_val=state.non_zero_val * weight,
        )

    def _adag_qi(self, state, ind):
        indices, _weight = mlx_create_c.creation_operator(state.non_zero_ind, ind, state.n, 3)
        ns = mlx_create_c.create_bos_ns(state.n, 3)
        exact_weight = np.asarray(
            [qi_sqrt_int(ns[index, ind] + 1) for index in state.non_zero_ind],
            dtype=object,
        )
        return self.transient_state_class(
            n=state.n + 1,
            non_zero_ind=indices,
            non_zero_val=np.asarray(
                [value * coeff for value, coeff in zip(state.non_zero_val, exact_weight)],
                dtype=object,
            ),
        )

    def _define_lambdas(self):
        self._axdag = lambda state: self._adag(state, 0)
        self._aydag = lambda state: self._adag(state, 1)
        self._azdag = lambda state: self._adag(state, 2)

        self._axdag_sq = lambda state: self._axdag(self._axdag(state))
        self._aydag_sq = lambda state: self._aydag(self._aydag(state))
        self._azdag_sq = lambda state: self._azdag(self._azdag(state))


class SphericalHOBasisNP(SphericalHOBasis):
    def __init__(self, Nmax):
        super().__init__(Nmax, exact=False)


SphericalHOBasisState = SphericalHOBasisStateQI
