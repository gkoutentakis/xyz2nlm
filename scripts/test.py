import numpy as np
import xyz2nlm
from xyz2nlm.angular_momentum_basis_tools import OddAngularMomentumBasis, EvenAngularMomentumBasis
from xyz2nlm.qi_interface import to_complex_array

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
    shob = xyz2nlm.create_spherical_ho_basis(5, method="Serial")
    am = xyz2nlm.create_angular_momentum_matrices(5, compatible_basis=shob)

    n = 3 
    view = shob.basis_states_views_n[n]
    am_ind = EvenAngularMomentumBasis(n) if n&1 == 0 else OddAngularMomentumBasis(n)

    # Print latex matrix with exact values
    print(f"Latex matrix of C^(n, l, m)_(n_x, n_y, n_z) for N={n}:\n")
    latex_print_QI_array(view)

    # convert to complex with the to_complex_array function
    print(f"\nNumpy matrix of C^(n, l, m)_(n_x, n_y, n_z) for N={n}:\n")
    print(to_complex_array(view))

    #print_the exact matrix elements of specific vector
    l, m = 3, 0
    print(f"\nPrinting the components of the vector |{n}, {l}, {m}>:")
    vector = view[:, am_ind.jm2index(l, m)]
    print("\n".join(str(vector[i]) for i in range(view.shape[0])))
            
if __name__ == "__main__":
    main()
