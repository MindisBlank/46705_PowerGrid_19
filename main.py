def main():
    import numpy as np
    import LoadNetworkData as ld
    from PowerFlow_46705 import PowerFlowNewton, DisplayResults

    # Load network data from file
    lnd = ld.load_network_data('testsystem.txt')
    (Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels, S_LD,
     MVA_base, V0, pq_index, pv_index, ref) = lnd

    # For testing, assume Sbus = -S_LD (i.e. injections are the negative of the loads)
    # This means that generation is determined by the slack bus.
    Sbus = -S_LD

    # Set maximum number of iterations and tolerance for convergence
    max_iter = 20
    err_tol = 1e-6

    # Run the Newton-Raphson power flow calculation
    V, success, n_iter = PowerFlowNewton(Ybus, Sbus, V0, pv_index, pq_index,
                                         max_iter, err_tol, print_progress=True)
    
    # Display the results (bus voltages, injections, branch flows, etc.)
    DisplayResults(V, lnd)


if __name__ == '__main__':
    main()
