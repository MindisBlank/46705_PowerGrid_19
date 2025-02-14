def main():
    import numpy as np
    import LoadNetworkData as ld
    from PowerFlow_46705 import PowerFlowNewton, DisplayResults
    

    # Load network data from file
    #lnd = ld.load_network_data('testsystem.txt')
    lnd = ld.load_network_data('Kundur_two_area_system.txt',debug=True)
    (Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels, Sbus, S_LD,MVA_base, V0, pq_index, pv_index, ref) = lnd

    # Set maximum number of iterations and tolerance for convergence
    max_iter = 30
    err_tol = 1e-4

    # Run the Newton-Raphson power flow calculation
    V, success, n_iter = PowerFlowNewton(Ybus,Sbus, V0, pv_index, pq_index,
                                         max_iter, err_tol, print_progress=True, debug=True)
    
    # Display the results (bus voltages, injections, branch flows, etc.)
    DisplayResults(V, lnd)

if __name__ == '__main__':
    main()
