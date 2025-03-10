import LoadNetworkData as ld
from PowerFlow_46705 import PowerFlowNewton, DisplayResults, DisplayResults_and_loading
from Contingency import system_violations,apply_contingency_to_Y_matrices
import numpy as np

def main():
    # Load network data from file
    #lnd = ld.load_network_data('testsystem.txt')
    #lnd = ld.load_network_data('Nordic32_SA.txt')
    #lnd=ld.load_network_data('TestSystem_with_trf.txt')
    lnd=ld.load_network_data('Kundur_two_area_system.txt')

    # Set maximum number of iterations and tolerance for convergence
    max_iter = 30
    err_tol = 1e-6

    # Run the Newton-Raphson power flow calculation
    V, success, n_iter = PowerFlowNewton(lnd.Ybus, lnd.Sbus, lnd.V0, lnd.pv_index, lnd.pq_index,
                                         max_iter, err_tol, print_progress=True)
    
    if success:
        # Display the results (bus voltages, injections, branch flows, etc.)
        DisplayResults_and_loading(V, lnd)
        
        # Now test the System_violations function
        violations = system_violations(V, lnd.Ybus, lnd.Y_from, lnd.Y_to, lnd)
        if violations:
            print("\nViolations detected:")
            for v in violations:
                print(v)
        else:
            print("\nNo violations detected in the system.")
    else:
        print("Power flow did not converge. Cannot check for violations.")

    # -------------------------------
    # Now test the contingency function.
    # For testing, select branch index 0.
    br_ind = 0
    fr_ind = lnd.branch_from[br_ind]   # the from-bus index for branch 0

    to_ind = lnd.branch_to[br_ind]   # the to-bus index for branch 0

    # Create a dummy 2x2 branch admittance matrix.
    # In practice, you would compute this from the branch's parameters.
    # For example, if the branch impedance gives an admittance of y, then:
    y = 0.1 + 1j*0.2
    Ybr_mat = np.array([[y, -y],
                        [-y, y]])

    # Apply the contingency (branch tripping)
    Ybus_mod, Y_fr_mod, Y_to_mod = apply_contingency_to_Y_matrices(lnd.Ybus, lnd.Y_from, lnd.Y_to,
                                                                   fr_ind, to_ind, br_ind, Ybr_mat)

    # print("\nModified Ybus matrix:")
    # print(Ybus_mod)
    # print("\nModified Y_fr matrix:")
    # print(Y_fr_mod)
    # print("\nModified Y_to matrix:")
    # print(Y_to_mod)


    # Display the results (bus voltages, injections, branch flows, etc.)
    #DisplayResults(V, lnd)
if __name__ == '__main__':
    main()