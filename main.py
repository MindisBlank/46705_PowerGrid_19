import LoadNetworkData as ld
from PowerFlow_46705 import PowerFlowNewton, DisplayResults, DisplayResults_and_loading
from Contingency import system_violations, apply_contingency_to_Y_matrices
import numpy as np

def main():
    # Load network data from file
    #lnd = ld.load_network_data('testsystem.txt')
    lnd = ld.load_network_data('Nordic32_SA.txt')
    # lnd = ld.load_network_data('TestSystem_with_trf.txt')

    # Set maximum number of iterations and tolerance for convergence
    max_iter = 30
    err_tol = 1e-6
    
    # Run the base-case Newton-Raphson power flow calculation
    V, success, n_iter = PowerFlowNewton(lnd.Ybus, lnd.Sbus, lnd.V0, lnd.pv_index, lnd.pq_index,
                                         max_iter, err_tol, print_progress=True)
    
    if success:
        # Display the base-case results (bus voltages, injections, branch flows, etc.)
        #DisplayResults_and_loading(V, lnd)
        
        # Check for any base-case violations
        violations = system_violations(V, lnd.Ybus, lnd.Y_from, lnd.Y_to, lnd)
        if violations:
            print("\nBase-case violations detected:")
            for v in violations:
                print(v)
        else:
            print("\nNo violations detected in the base-case system.")
    else:
        print("Base-case power flow did not converge. Cannot check for violations.")


    # N-1 Branch Contingency Analysis
    print("\n=============================")
    print("Starting N-1 Branch Contingency Analysis")
    print("=============================")

    # Loop over each branch to simulate its outage
    num_branches = len(lnd.branch_from)
    for br_ind in range(num_branches):
        print(f"\n--- Simulating branch contingency for branch index {br_ind} "
            f"(from bus {lnd.branch_from[br_ind]} to bus {lnd.branch_to[br_ind]}) ---")
        
        # Create a dummy branch admittance matrix (replace with actual branch parameters as needed)
        y = 0.1 + 1j*0.2
        Ybr_mat = np.array([[y, -y],
                            [-y, y]])
        
        # Get the from and to bus indices for this branch
        fr_ind = lnd.branch_from[br_ind]
        to_ind = lnd.branch_to[br_ind]
        
        # Apply the contingency: trip the branch using the contingency function
        Ybus_mod, Y_fr_mod, Y_to_mod = apply_contingency_to_Y_matrices(lnd.Ybus, lnd.Y_from, lnd.Y_to,
                                                                    fr_ind, to_ind, br_ind, Ybr_mat)
        
        # Run power flow with the modified admittance matrix
        V_mod, success_mod, n_iter_mod = PowerFlowNewton(Ybus_mod, lnd.Sbus, lnd.V0,
                                                        lnd.pv_index, lnd.pq_index,
                                                        max_iter, err_tol, print_progress=False)
        if success_mod:
            print("Power flow converged for branch contingency.")
            # Optionally, display detailed results for the contingency case.
            #DisplayResults(V_mod, lnd)
            
            # Check for system violations using the modified admittance matrices
            violations = system_violations(V_mod, Ybus_mod, Y_fr_mod, Y_to_mod, lnd)
            if violations:
                print("Violations detected:")
                for v in violations:
                    print(v)
            else:
                print("No violations detected.")
        else:
            print("Power flow did not converge for branch contingency.")

if __name__ == '__main__':
    main()
