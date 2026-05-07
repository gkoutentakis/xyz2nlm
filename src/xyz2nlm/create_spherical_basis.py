import numpy as np
import numpy.testing as npt
import mlx_create_c
import concurrent.futures as cf
from os import cpu_count

from scipy.sparse.linalg import LinearOperator, cg

from .custom_dataclasses import TransientState, SphericalHOBasisState
from .create_angular_momentum_matrices import CreateAngularMomentumMatrices
from ._angular_momentum_basis_tools import EvenAngularMomentumBasis, OddAngularMomentumBasis
from .direct_seed_state_calculation import create_seed_state

class BlockLmApplication:
    def __init__(self, n, block_in, ls, ms):
        self.am = CreateAngularMomentumMatrices(n)
        _, self.Lm = self.am.get_ladder_operators(operator_buffer_size=self.am.basis_size)

        self.work_array_a = block_in
        self.work_array_b = self.am._Lm_buffer

        self.states_current = self.work_array_a
        self.states_current_using_buffer_a = True

        self.block_ls = ls.copy()
        self.block_ms = ms.copy()

    def decrease_m_of_block(self):
        self.states_current = (self.Lm @ self.states_current) # states_current -> Lm buffer
        self.update_working_buffer_data()
        self.normalize_states_after_lm_application()

        self.block_ms -= 1 # update ms

        return self.states_current, self.block_ls, self.block_ms

    def update_working_buffer_data(self):
        if self.states_current_using_buffer_a:
            self.am._change_buffer_of_operator("Lm", self.work_array_a)
            self.states_current_using_buffer_a = False
        else:
            self.am._change_buffer_of_operator("Lm", self.work_array_b)
            self.states_current_using_buffer_a = True

    def normalize_states_after_lm_application(self):
        self.states_current /= np.sqrt((self.block_ls + self.block_ms) *
                                       (self.block_ls - self.block_ms + 1))[None, :] # normalize

    def drop_first_state_from_block(self):
        self.states_current = self.states_current[:, 1:]
        self.block_ls = self.block_ls[1:]
        self.block_ms = self.block_ms[1:]



class SphericalHOBasis:

    def __init__(self, Nmax):
        self.Nmax = Nmax
        self._define_lambdas()
        self._set_basis_state_matrices()
        self.initialize_calculation()

    # initial states
    def _initial_state_even_sector(self) -> SphericalHOBasisState:
        return SphericalHOBasisState(n=0, l=0, m=0,
                                     non_zero_ind=np.array([0]),
                                     non_zero_val=np.array([1.]))

    def _initial_state_odd_sector(self) -> SphericalHOBasisState:
        return SphericalHOBasisState(n=1, l=1, m=1,
                                     non_zero_ind=np.array([0, 1]),
                                     non_zero_val=np.array([-1./np.sqrt(2), -1j/np.sqrt(2)]))

    def initialize_calculation(self):
        self.seed_states = {}

        # index 1 = n index2 = l
        n = np.arange(self.Nmax+1)[:, None]
        m = np.arange(self.Nmax+1)[None,:]

        self.pending_seed_state_checklist = n >= m          # n >=l
        self.pending_seed_state_checklist *= (n + m) % 2 == 0 # n, l have the same parity

        self.available_seed_state_checklist = np.zeros_like(self.pending_seed_state_checklist)

        self.record_seed_state(self._initial_state_even_sector())
        self.record_seed_state(self._initial_state_odd_sector())

        self.pending_blocks_checklist = np.ones((self.Nmax+1,), dtype=bool)
        self.running_blocks_checklist = np.zeros_like(self.pending_blocks_checklist)
        self.available_blocks_checklist = np.zeros_like(self.pending_blocks_checklist)
        self.update_block_availability()

        self.jobs_dispatched = {}

    def calculate_states(self, method="Threads", max_workers=None):
        if method=="Threads":
            Executor = cf.ThreadPoolExecutor
        elif method=="Serial":
            Executor = cf.ThreadPoolExecutor
            max_workers = 1
        else:
            raise ValueError(f"Unknown method: {method}\n"
                             "Use either 'Threads', 'Serial'")

        proposed_workers = (cpu_count() or 1)
        self.workers = proposed_workers if (max_workers is None) \
            or (proposed_workers <= max_workers) else max_workers

        self.pool = Executor(max_workers=self.workers)

        while not self.calculation_done():
            while len(self.jobs_dispatched) < self.workers:
                if not self.dispatch_job():
                    break

            done, _ = cf.wait(self.jobs_dispatched.keys(),
                              return_when=cf.FIRST_COMPLETED)

            for future in done:
                self.process_result(future)
            self.update_block_availability()

    def process_result(self, future):
        job_type = self.jobs_dispatched.pop(future)

        if job_type == "Block":
            self.record_block(future.result())
            # print(f"Block: {future.result()} done")
            return 

        if job_type == "Seed State":
            self.record_seed_state(future.result())
            # print(f"Seed State: {future.result().n}, {future.result().l} done")
            return

        raise ValueError(f"Unknown job type {job_type}")

    def dispatch_job(self):
        if (n := self.next_available_block()) is not None:
            self.available_blocks_checklist[n] = False
            self.running_blocks_checklist[n] = True
            future = self.pool.submit(self.job_make_block, n)
            self.jobs_dispatched[future] = "Block"
            # print(f"Block: {n} dispatched")
            return True

        if (shape := self.next_available_seed_state()) is not None:
            n, l = shape
            self.available_seed_state_checklist[n, l] = False
            future = self.pool.submit(self.job_make_seed_state, n, l)
            self.jobs_dispatched[future] = "Seed State"
            # print(f"Seed State: {n}, {l} dispatched")
            return True

        return False

    def job_make_block(self, n):
        seed_states = [self.seed_states[(n, m)] for m in range(0 if n%2==0 else 1, n+1, 2)]
        self.complete_fixed_n_block_from_seed_states(seed_states)
        return n

    def record_block(self, n):
        self.pending_blocks_checklist[n] = False

    def job_make_seed_state(self, n, l):
        return create_seed_state(n, l)
        # if l != n:
        #     seed = self.seed_states[(n-2, l)]
        #     result = self.s2_operator(seed)
        # else:
        #     seed = self.seed_states[(n-2, l-2)]
        #     result = self.r22_operator(seed)
        #     # try to refine the seed state
        #     # seed_array = result.as_ndarray()
        #     # seed_array /= np.linalg.norm(seed_array)
        #     # x = seed_array

        #     # tau = 1e-10
        #     # self.Lp, _ = CreateAngularMomentumMatrices(result.n).get_ladder_operators()

        #     # Lp = self.Lp @ np.eye(self.Lp.shape[0], dtype=self.Lp.dtype)
        #     # hpsi = lambda psi: Lp.conj().T @ (Lp @ psi) + tau * psi
        #     # mat = LinearOperator(Lp.shape, matvec=hpsi, dtype=Lp.dtype)

        #     # assert Lp.shape[0] == seed_array.shape[0]

        #     # for i in range(20):
        #     #     y, _ = cg(mat, x, rtol=1e-12, atol=0)
        #     #     x = y/np.linalg.norm(y)

        #     # result = SphericalHOBasisState(n=result.n, l=result.l, m=result.m,
        #     #                                non_zero_ind=np.nonzero(x)[0],
        #     #                                non_zero_val=x[x!=0])


        # return result

    def record_seed_state(self, seed_state):
        n, l, m = seed_state.get_indices()

        self.seed_states[(n, m)] = seed_state
        self.pending_seed_state_checklist[n, m] = False

        if n+2 > self.Nmax:
            return 

        if l == n:
            self.available_seed_state_checklist[n+2, l+2] = True

        self.available_seed_state_checklist[n+2, l] = True

    def next_available_seed_state(self):
        ind = np.flatnonzero(self.available_seed_state_checklist)

        if ind.shape[0] > 0:
            flat_ind = ind[0]
            return np.unravel_index(flat_ind, self.available_seed_state_checklist.shape)
        else:
            return None

    def next_available_block(self):
        ind = np.flatnonzero(self.available_blocks_checklist)

        if ind.shape[0] > 0:
            return ind[0]
        else:
            return None

    def update_block_availability(self):
        completed_seed_state_blocks = np.sum(self.pending_seed_state_checklist, axis=1) == 0
        self.available_blocks_checklist[completed_seed_state_blocks] = True
        self.available_blocks_checklist *= self.pending_blocks_checklist
        self.available_blocks_checklist *= np.logical_not(self.running_blocks_checklist)

    def calculation_done(self):
        return np.sum(self.pending_blocks_checklist) == 0

    # orchestration methods
    def pack_list_spherical_basis_states(self,
                                         states: list[SphericalHOBasisState]) -> tuple[int, np.ndarray]:
        indices = np.array([list(state.get_indices()) for state in states])

        #check indices
        n = indices[0, 0]
        assert indices.shape[0] == n//2+1 if n%2==0 else (n+1)//2
        npt.assert_array_equal(indices[:, 0], n)
        npt.assert_array_equal(indices[:, 1], indices[:, 2]) # to proceed l == m

        order = np.argsort(indices[:, 1]) # order l's
        assert indices[order[-1], 1] == n

        out = np.column_stack([state.as_ndarray() for state in states])
        return n, out[:, order]

    def _set_basis_state_matrices(self):
        elements_sector_n = lambda n: ((n+1) * (n+2))//2
        number_of_elements_to_n = lambda n:\
            ((n + 1) * (n + 2) * (n + 3) * (3 * n**2 + 12 * n + 10))//60 \
            if n >=0 else 0

        self.basis_states = np.zeros((number_of_elements_to_n(self.Nmax),), dtype=complex)
        self.basis_states_views_n = \
            [self.basis_states[number_of_elements_to_n(i-1):number_of_elements_to_n(i)]\
             .reshape((elements_sector_n(i), elements_sector_n(i)), copy=False)\
             for i in range(self.Nmax+1)]

    # calculation methods
    def s2_operator(self, state_in: SphericalHOBasisState) -> SphericalHOBasisState:
        n, l, m = state_in.get_indices()

        n_out = n + 2
        l_out = l
        m_out = m

        # normalization change due to the application of the operator
        overall_weight = ((n - l + 2) * (n + l +3))**(-0.5)

        termx = self._axdag_sq(state_in)
        termy = self._aydag_sq(state_in)
        termz = self._azdag_sq(state_in)

        total_state = (termx + termy + termz).scale(overall_weight)


        return SphericalHOBasisState(n=n_out, l=l_out, m=m_out,
                                     non_zero_ind=total_state.non_zero_ind,
                                     non_zero_val=total_state.non_zero_val)

    def r22_operator(self, state_in: SphericalHOBasisState) -> SphericalHOBasisState:
        n, l, m = state_in.get_indices() 
        n_out = n + 2
        l_out = l + 2
        m_out = m + 2

        overall_weight = 0.5 * np.sqrt((2 * l + 3) * (2 * l + 5) /
                                       ((l  + 1) * (l + 2) * (n + l + 3) * (n + l + 5))) 

        termx = self._axdag_sq(state_in)
        termy = self._aydag_sq(state_in).scale(-1)
        termxy = self._axdag(self._aydag(state_in)).scale(2j)

        total_state = (termx + termy + termxy).scale(overall_weight)

        return SphericalHOBasisState(n=n_out, l=l_out, m=m_out,
                                     non_zero_ind=total_state.non_zero_ind,
                                     non_zero_val=total_state.non_zero_val)


    def complete_fixed_n_block_from_seed_states(self, seed_states: list[SphericalHOBasisState]):

        n, block_in = self.pack_list_spherical_basis_states(seed_states)
        if n == 0:
            self.basis_states_views_n[0][0] = 1.0
            return

        block_out_view = self.basis_states_views_n[n]

        if n%2==0:
            ind = EvenAngularMomentumBasis(n)
            # the l = 0, m = 0 case is trivial do it immediately and proceed with l > 0
            block_out_view[:, ind.jm2index(0, 0)] = block_in[:, 0]
            block_in = block_in[:, 1:]
            ls = np.arange(2, n+1, 2)
        else:
            ind = OddAngularMomentumBasis(n)
            ls = np.arange(1, n+1, 2)

        block_out_view[:, ind.jm2index(ls, ls)] = block_in
        view_pos = id(block_out_view)

        #call the controller
        block_controller = BlockLmApplication(n, block_in, ls, ls)
        for first_l_in_block in ls:
            for j in range(4 if first_l_in_block != 1 else 2):
                states_current, ls_c, ms_c = block_controller.decrease_m_of_block()
                block_out_view[:, ind.jm2index(ls_c, ms_c)] = states_current
                assert id(block_out_view) == view_pos

            # drop first state as its m cannot be reduced further
            block_controller.drop_first_state_from_block()

    # helper methods
    def _adag(self, state, ind):
        indices, weight = mlx_create_c.creation_operator(state.non_zero_ind, ind, state.n, 3)
        return TransientState(n=state.n+1,
                              non_zero_ind=indices,
                              non_zero_val=state.non_zero_val * weight)

    def _define_lambdas(self):
        self._axdag = lambda state: self._adag(state, 0)
        self._aydag = lambda state: self._adag(state, 1)
        self._azdag = lambda state: self._adag(state, 2)

        self._axdag_sq = lambda state: self._axdag(self._axdag(state))
        self._aydag_sq = lambda state: self._aydag(self._aydag(state))
        self._azdag_sq = lambda state: self._azdag(self._azdag(state))
