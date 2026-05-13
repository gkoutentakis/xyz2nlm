import numpy as np
import xyz2nlm
from xyz2nlm.angular_momentum_basis_tools import OddAngularMomentumBasis, EvenAngularMomentumBasis
from xyz2nlm.qi_interface import to_complex_array, exact_inner_product
from mlx_create_c import create_bos_ns

import matplotlib.pyplot as plt
import scipy.sparse

def latex_print_QI_array(array):
    string = r"\begin{equation}"+ "\n" + r"\left[" + \
        r"\begin{array}{" + (" c " * array.shape[1]) + "}\n"
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            string += array[i, j].print(mode="latex") + (
                "&" if j != array.shape[1] -1 else "")
        string += (r"\\" if i != array.shape[0] - 1 else "") + "\n"
    string += r"\end{array} \right]" + "\n" + r"\end{equation}"
    print(string)


def main():
    print("\n------------------------------------------------------------------")
    print(  "-----------------  xyz2nlm usage tutorial  -----------------------")
    print(  "------------------------------------------------------------------")

    # The command below calculates the transformation Cartesian HG -> Spherical LG
    # exact -> True uses the QINumber exact approach.
    # exact -> False uses the complex double precision approach.
    # for exact -> True use the method -> "Serial" since QINumber are python objects
    #                   their implementation does not allow for parallelization due
    #                   to GIL restrictions.
    # for exact -> False prefer method -> "Threads" since the parallelization makes
    #                    it faster.
    shob = xyz2nlm.create_spherical_ho_basis(5, exact=True, method="Serial")

    # here we select the N we want to work with
    n = 3
    view = shob.basis_states_views_n[n]
    # The data in the numpy array view are organized in terms of:
    # rows -> n_x, n_y, n_z configurations with N = n_x + n_y + n_z = constant
    # columns -> N=fixed, L, M configurations
    # the row indices can be obtained with the mlx_create_c.create_bos_ns method
    # the column indices can be obtained with the Even/OddAngualarMomentumBasis
    ca_ind = create_bos_ns(n, 3)
    am_ind = EvenAngularMomentumBasis(n) if n&1 == 0 else OddAngularMomentumBasis(n)

    # Print latex matrix with exact values
    print(f"\nLatex matrix of C^(n, l, m)_(n_x, n_y, n_z) for N={n}:\n")
    latex_print_QI_array(view)

    # convert to complex with the to_complex_array function
    print(f"\nNumpy matrix of C^(n, l, m)_(n_x, n_y, n_z) for N={n}:\n")
    print(to_complex_array(view))

    #print_the exact matrix elements of specific vector with the associated
    #Cartesian components
    l, m = 3, 0
    print(f"\nPrinting the components of the vector |{n}, {l}, {m}>:")
    print("In the format: < n_x, n_y, n_z | N, L, M >")
    vector = view[:, am_ind.jm2index(l, m)]
    confs = [f"< {n_x}, {n_y}, {n_z} | {n}, {l}, {m} > =" for  n_x, n_y, n_z in ca_ind]
    print("\n".join(("\n".join((confs[i], str(vector[i]))) for i in range(view.shape[0]))))

    # We can do operations with the QINumber vectors by using the appropriate arrays
    am = xyz2nlm.create_angular_momentum_matrices(n, compatible_basis=shob)

    # We could also initialize as follows but passing the basis is safer
    # am = xyz2nlm.create_angular_momentum_matrices(n, exact=True)
    # also the operators for complex numbers with the same interface are available in
    # am = xyz2nlm.create_angular_momentum_matrices(n, exact=False)

    Lx, Ly, Lz = am.get_linear_operators()
    Lp, Lm = am.get_ladder_operators()
    L2 = am.get_casimir()

    # The same vector after Lz is applied
    vector_new = Lz @ vector

    print(f"\nPrinting the components of the vector |Psi > = L_z |{n}, {l}, {m}>:")
    print("In the format: < n_x, n_y, n_z | Psi >")
    confs = [f"< {n_x}, {n_y}, {n_z} | {n}, {l}, {m} > =" for  n_x, n_y, n_z in ca_ind]
    print("\n".join(("\n".join((confs[i], str(vector_new[i]))) for i in range(view.shape[0]))))

    # to calculate matrix elements of vectors we use the exact_inner_product method
    vector_new = L2 @ vector
    L2_val = exact_inner_product(vector, vector_new)
    print(f"\nThe expectation value of <{n}, {l}, {m}| L^2 |{n}, {l}, {m}> = {L2_val}")

    print("\nWe convert to double complex and do the same")
    vector_complex = to_complex_array(vector)

    am = xyz2nlm.create_angular_momentum_matrices(n, exact=False)

    Lx, Ly, Lz = am.get_linear_operators()
    Lp, Lm = am.get_ladder_operators()
    L2 = am.get_casimir()

    vector_new = Lz @ vector_complex
    print(f"\nPrinting the components of the vector |Psi > = L_z |{n}, {l}, {m}>:")
    print("In the format: < n_x, n_y, n_z | Psi >")
    confs = [f"< {n_x}, {n_y}, {n_z} | {n}, {l}, {m} > =" for  n_x, n_y, n_z in ca_ind]
    print("\n".join((" ".join((confs[i], str(vector_new[i]))) for i in range(view.shape[0]))))

    vector_new = L2 @ vector_complex
    L2_val = np.conj(vector_complex).T @ vector_new
    print(f"\nThe expectation value of <{n}, {l}, {m}| L^2 |{n}, {l}, {m}> = {L2_val}")

    print("\nFor low enough N values the results should approximately the same")

if __name__ == "__main__":
    main()
