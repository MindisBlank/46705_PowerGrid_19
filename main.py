
def main():
    import LoadNetworkData as ld
    from PowerFlow_46705 import PowerFlowNewton, DisplayResults, DisplayResults_and_loading
    

    # Load network data from file
    #lnd = ld.load_network_data('testsystem.txt')
    lnd = ld.load_network_data('Nordic32_SA.txt')
    #lnd=ld.load_network_data('TestSystem_with_trf.txt')
    (Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels, Sbus, S_LD,
            MVA_base, V0, pq_index, pv_index, ref, Gen_rating,Br_rating,BUS_NR,FROM_BUS_AND_TO_BUS,Tran_rating) = lnd
    
    #print("Ybus:", Ybus)
    # print("Y_fr:", Y_fr)
    # print("Y_to:", Y_to)
    # print("br_f:", br_f)
    # print("br_t:", br_t)
    # print("buscode:", buscode)
    # print("bus_labels:", bus_labels)
    # print("Sbus:", Sbus)
    # print("S_LD:", S_LD)
    # print("MVA_base:", MVA_base)
    # print("V0:", V0)
    # print("pq_index:", pq_index)
    # print("pv_index:", pv_index)
    # print("ref:", ref)
    # print("Gen_rating:", Gen_rating)
    # print("Br_rating:", Br_rating)
    # Set maximum number of iterations and tolerance for convergence
    max_iter = 30
    err_tol = 1e-4

    # Run the Newton-Raphson power flow calculation
    V, success, n_iter = PowerFlowNewton(Ybus,Sbus, V0, pv_index, pq_index,
                                         max_iter, err_tol, print_progress=True)
    
    # Display the results (bus voltages, injections, branch flows, etc.)
    #DisplayResults(V, lnd)

    DisplayResults_and_loading(V, lnd)

if __name__ == '__main__':
    main()