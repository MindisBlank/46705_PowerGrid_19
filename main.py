import LoadNetworkData as ld
from PowerFlow_46705 import PowerFlowNewton, DisplayResults, DisplayResults_and_loading
from Contingency import system_violations, apply_contingency_to_Y_matrices
import numpy as np

def main():
    # Load network data from file
    lnd = ld.load_network_data('Nordic32_SA.txt')

    # Set maximum number of iterations and tolerance for convergence
    max_iter = 30
    err_tol = 1e-6

    # Run the Newton-Raphson power flow calculation
    V, success, n_iter = PowerFlowNewton(lnd.Ybus, lnd.Sbus, lnd.V0, lnd.pv_index, lnd.pq_index,
                                         max_iter, err_tol, print_progress=True)
    
    if success:
        # Display the results (bus voltages, injections, branch flows, etc.)
        DisplayResults_and_loading(V, lnd)
        
        # Check for generator overloads and bus voltage violations
        check_system_violations(V, lnd)
        
        # Run contingency analysis
        run_contingency_analysis(lnd, max_iter, err_tol)
    else:
        print("Power flow did not converge. Cannot check for violations.")

def check_system_violations(V, lnd):
    voltage_violations = []
    overloads = []

    # Check bus voltages
    for i, voltage in enumerate(V):
        if abs(voltage) < 0.95 or abs(voltage) > 1.05:
            voltage_violations.append((i, voltage))

    # Check generator overloads if gen_data attribute exists
    if hasattr(lnd, 'gen_data'):
        for i, gen in enumerate(lnd.gen_data):
            P_gen = gen['P_gen']
            Q_gen = gen['Q_gen']
            P_max = gen['P_max']
            Q_max = gen['Q_max']
            Q_min = gen['Q_min']
            if P_gen > P_max or Q_gen > Q_max or Q_gen < Q_min:
                overloads.append((i, P_gen, Q_gen))

    # Display violations
    if voltage_violations:
        print("\nBus Voltage Violations:")
        for bus, voltage in voltage_violations:
            print(f"Bus {bus}: Voltage = {voltage:.4f} pu")
    else:
        print("\nNo bus voltage violations detected.")

    if overloads:
        print("\nGenerator Overloads:")
        for gen, P_gen, Q_gen in overloads:
            print(f"Generator {gen}: P_gen = {P_gen:.4f} MW, Q_gen = {Q_gen:.4f} MVar")
    else:
        print("\nNo generator overloads detected.")

def run_contingency_analysis(lnd, max_iter, err_tol):
    for br_ind in range(len(lnd.branch_from)):
        fr_ind = lnd.branch_from[br_ind]   # the from-bus index for the branch
        to_ind = lnd.branch_to[br_ind]     # the to-bus index for the branch

        # Get the branch parameters (r, x, b) from the network data
        r = lnd.branch_rating[br_ind][2]
        x = lnd.branch_rating[br_ind][3]
        b = lnd.branch_rating[br_ind][4]
        y = 1 / complex(r, x)  # Calculate the admittance of the branch

        # Create the branch admittance matrix
        Ybr_mat = np.array([[y + 1j * b / 2, -y],
                            [-y, y + 1j * b / 2]])

        # Apply the contingency (branch tripping)
        Ybus_mod, Y_fr_mod, Y_to_mod = apply_contingency_to_Y_matrices(lnd.Ybus, lnd.Y_from, lnd.Y_to,
                                                                       fr_ind, to_ind, br_ind, Ybr_mat)

        # Run the Newton-Raphson power flow calculation with the modified Ybus
        V, success, n_iter = PowerFlowNewton(Ybus_mod, lnd.Sbus, lnd.V0, lnd.pv_index, lnd.pq_index,
                                             max_iter, err_tol, print_progress=False)
        
        if success:
            # Check for violations
            violations = system_violations(V, Ybus_mod, Y_fr_mod, Y_to_mod, lnd)
            if violations:
                print(f"\nContingency {br_ind}: Violations detected:")
                for v in violations:
                    print(v)
            else:
                print(f"\nContingency {br_ind}: No violations detected.")
        else:
            print(f"\nContingency {br_ind}: Power flow did not converge.")

if __name__ == '__main__':
    main()