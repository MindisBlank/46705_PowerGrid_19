import os
import LoadNetworkData as ld
from PowerFlow_46705 import PowerFlowNewton, DisplayResults, DisplayResults_and_loading
from Contingency import system_violations, apply_contingency_to_Y_matrices
import numpy as np

def main():
    # Create the "results" folder if it does not exist.
    os.makedirs("results", exist_ok=True)
    results_file = os.path.join("results", "branch_contingency_results.txt")
    
    with open(results_file, "w") as out:
        # Load network data from file
        lnd = ld.load_network_data('Nordic32_SA.txt')

        # max iterations and tolerance
        max_iter = 30
        err_tol = 1e-4
        
        #base-case
        V, success, n_iter = PowerFlowNewton(
            lnd.Ybus, lnd.Sbus, lnd.V0, 
            lnd.pv_index, lnd.pq_index,
            max_iter, err_tol, print_progress=True
        )
        
        out.write("********** Base-case Results **********\n")
        if success:
            out.write("Base-case power flow converged.\n")
            # Check for base-case violations
            #DisplayResults_and_loading(V, lnd)
            violations = system_violations(V, lnd.Ybus, lnd.Y_from, lnd.Y_to, lnd)
            if violations:
                out.write("Base-case violations detected:\n")
                for v in violations:
                    out.write(str(v) + "\n")
            else:
                out.write("No violations detected in the base-case system.\n")
        else:
            out.write("Base-case power flow did not converge. Cannot check for violations.\n")
        
        out.write("\n=============================\n")
        out.write("Starting N-1 Branch Contingency Analysis\n")
        out.write("=============================\n")
    
        # Loop over each branch to simulate its outage
        num_branches = len(lnd.branch_from)
        for br_ind in range(num_branches):
            out.write(f"\n--- Simulating branch outage for index {br_ind} "
                    f"(from bus {lnd.branch_from[br_ind]} to bus {lnd.branch_to[br_ind]}) ---\n")
            
            fr_ind = lnd.branch_from[br_ind]
            to_ind = lnd.branch_to[br_ind]
            
            # Compute branch admittance matrix from stored values:
            Ybr_mat = np.array([[lnd.Y_from[br_ind, fr_ind], lnd.Y_from[br_ind, to_ind]],
                                [lnd.Y_to[br_ind, fr_ind], lnd.Y_to[br_ind, to_ind]]])
            
            # Remove the branch
            Ybus_mod, Y_fr_mod, Y_to_mod = apply_contingency_to_Y_matrices(
                lnd.Ybus, lnd.Y_from, lnd.Y_to,
                fr_ind, to_ind, br_ind, Ybr_mat
            )
            
            # Run power flow on the modified network
            try:
                V_mod, success_mod, n_iter_mod = PowerFlowNewton(
                    Ybus_mod, lnd.Sbus, lnd.V0,
                    lnd.pv_index, lnd.pq_index,
                    max_iter, err_tol, print_progress=False
                )
            except np.linalg.LinAlgError:
                out.write("Power flow did not converge due to singular matrix (system may be disconnected).\n")
                continue

            if success_mod:
                out.write("Power flow converged for this branch outage.\n")
                violations = system_violations(V_mod, Ybus_mod, Y_fr_mod, Y_to_mod, lnd)
                if violations:
                    out.write("Violations detected:\n")
                    for v in violations:
                        out.write(str(v) + "\n")
                else:
                    out.write("No violations detected.\n")
            else:
                out.write("Power flow did not converge for this branch outage.\n")

    
    print(f"Results saved in {results_file}")

if __name__ == '__main__':
    main()
