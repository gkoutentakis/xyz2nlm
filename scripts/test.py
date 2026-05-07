import numpy as np
import xyz2nlm

import matplotlib.pyplot as plt
import scipy.sparse

from xyz2nlm.create_spherical_basis import BlockLmApplication
from xyz2nlm.direct_seed_state_calculation import create_seed_state

def main():


    shob = xyz2nlm.SphericalHOBasis(2)

    stat1 = shob._initial_state_even_sector()
    stat2 = shob.s2_operator(stat1)
    stat3 = shob.r22_operator(stat1)

    print("generated directlty")
    state = create_seed_state(2, 0)
    state.print_element()
    print("s2 operator")
    stat2.print_element()

    # print("generated directlty")
    # state = create_seed_state(10, 4)
    # state.print_element()
    
    # print("generated directlty")
    # state = create_seed_state(5, 3)
    # state.print_element()

    print("generated directlty")
    create_seed_state(2, 2).print_element()
    print("r22 operator")
    stat3.print_element()
    
    
def shob_test():
    shob = xyz2nlm.SphericalHOBasis(80)
    #shob.calculate_states(method="Serial")

    shob.calculate_states()
    # print(shob.basis_states_views_n[0])
    # print(shob.basis_states_views_n[1])
    # print(shob.basis_states_views_n[2])

    #print(shob.basis_states_views_n[3])

    view = shob.basis_states_views_n[80]

    print(np.sum(np.abs(view)**2, axis=0))
    print(np.count_nonzero(view)/np.prod(view.shape))
    plt.pcolormesh(np.abs(np.conj(view).T @ view))
    plt.show()


def many_stuff():

    shob = xyz2nlm.SphericalHOBasis(2)

    stat1 = shob._initial_state_even_sector()
    stat1a = shob._initial_state_odd_sector()
    stat2 = shob.s2_operator(stat1)
    stat3 = shob.r22_operator(stat1)

    print(f"Basis state: n={stat1.n} l={stat1.l}, m={stat1.m}")
    stat1.print_element()
    print(f"Basis state: n={stat2.n} l={stat2.l}, m={stat2.m}")
    stat2.print_element()
    print(f"Basis state: n={stat3.n} l={stat3.l}, m={stat3.m}")
    stat3.print_element()


    print(f"Basis state: n={stat1.n} l={stat1.l}, m={stat1.m}")
    stat1.print_element()

    shob._set_basis_state_matrices()
    print(shob.basis_states_views_n[0])
    print(shob.basis_states_views_n[1])
    print(shob.basis_states_views_n[2])
    print(np.prod(shob.basis_states_views_n[0].shape) + \
          np.prod(shob.basis_states_views_n[1].shape) + \
          np.prod(shob.basis_states_views_n[2].shape))
    print(shob.basis_states.shape)

    states_second = [stat2, stat3]

    print(stat2.as_ndarray())
    print(stat3.as_ndarray())
    n, block = shob.pack_list_spherical_basis_states(states_second)
    print(n)
    print(block)

    am = xyz2nlm.CreateAngularMomentumMatrices(2)
    lhs_map_term, rhs_map_term, coefficients_term = \
        am._calculate_linear_operator_elements(1, 2)
    print(lhs_map_term.shape, rhs_map_term.shape, coefficients_term.shape)

    shob.complete_fixed_n_block_from_seed_states([stat1])
    shob.complete_fixed_n_block_from_seed_states([stat1a])

    n, block_in = shob.pack_list_spherical_basis_states([stat1a])
    block_controller = BlockLmApplication(n, block_in, np.array([1]), np.array([1]))
    states_current, ls_c, ms_c = block_controller.decrease_m_of_block()
    print("manual application to L = 1")
    print(states_current)

    shob.complete_fixed_n_block_from_seed_states(states_second)
    print(shob.basis_states_views_n[0])
    print(shob.basis_states_views_n[1])
    print(shob.basis_states_views_n[2])

    Lx, Ly, Lz = am.get_linear_operators()
    L2 = am.get_casimir()

    bas = shob.basis_states_views_n[2]
    Lz_av = np.conjugate(bas).T @ (Lz @ bas)
    L2_av = np.conjugate(bas).T @ (L2 @ bas)
    print(np.hstack((np.diag(L2_av), np.diag(Lz_av))))
    
            
if __name__ == "__main__":
    main()
