
def apply_contingency_to_Y_matrices(Ybus , Yfr , Yto , fr_ind , to_ind , br_ind , Ybr_mat) :
    # input:
    # The original admitance matirces: Ybus,Yfr,Yto
    # The from and to end indices for the branch (fr_ind, to_ind)
    # The indice for where the branch is in the branch list (br_ind)
    # The 2x2 admittance matrix for the branch Ybr_mat
    ##########################################################

    # This is important, you must copy the original matrices
    Ybus_mod = Ybus . copy ( ) # otherwise you will change the original Ybus matrix
    Yfr_mod = Yfr . copy ( ) # when ever you make changes to Ybus_mod
    Yto_mod = Yto . copy ( ) # using the .copy() function avoids this

    ##################################################################################
    #
    # YOUR CODE COMES HERE:
    #
    # 1. Remove the branch from the Ybus_mod matrix
    # 2. Remove the branch from the Yto and Yfr matrices
    #
    ####################################################################################

    return Ybus_mod , Yfr_mod , Yto_mod




def System_violations(V, Ybus, Y_from, Y_to, lnd):
    # Inputs:
    # V = results from the load flow
    # Ybus = the bus admittance matrix used in the load flow
    # Y_from, Y_to = the admittance matrices used to determine the branch flows
    # lnd = the LoadNetworkData object for easy access to other model data

    # store variables as more convenient names
    br_f = lnd.br_f         # from-bus branch indices
    br_t = lnd.br_t         # to-bus branch indices
    ind_to_bus = lnd.ind_to_bus  # mapping from index to bus number
    bus_to_ind = lnd.bus_to_ind  # mapping from bus number to index
    br_MVA = lnd.br_MVA      # branch flow MVA ratings
    br_id = lnd.br_id        # branch identifiers

    # Calculate line flows and injections
    # The conjugate (conj) is used so that |S| = |V|*|I|
    S_to = V[br_t] * (Y_to.dot(V)).conj()    # flow into the 'to' end of each branch
    S_from = V[br_f] * (Y_from.dot(V)).conj()  # flow into the 'from' end of each branch
    S_inj = V * (Ybus.dot(V)).conj()           # net injection at each bus
    SLD = lnd.S_LD                           # defined loads at PQ buses
    S_gen = S_inj + SLD                       # generator outputs (net injections plus loads)

    violations = []  # empty list that will store strings describing each violation

    # 1. Check branch flows at both ends and report if the limits are violated
    for i in range(len(br_f)):
        flow_from = abs(S_from[i])
        flow_to = abs(S_to[i])
        limit = br_MVA[i]
        if flow_from > limit:
            violations.append("Branch {} (bus {} -> bus {}) from-end flow violation: {:.2f} MVA > limit {:.2f} MVA".format(
                br_id[i], lnd.ind_to_bus[br_f[i]], lnd.ind_to_bus[br_t[i]], flow_from, limit))
        if flow_to > limit:
            violations.append("Branch {} (bus {} -> bus {}) to-end flow violation: {:.2f} MVA > limit {:.2f} MVA".format(
                br_id[i], lnd.ind_to_bus[br_f[i]], lnd.ind_to_bus[br_t[i]], flow_to, limit))

    # 2. Check generator outputs if generator limits are defined in lnd.
    # Here we assume that if lnd has an attribute 'gen_MVA' and 'gen_bus', then
    # gen_MVA[i] is the limit for the generator at bus gen_bus[i].
    if hasattr(lnd, 'gen_MVA') and hasattr(lnd, 'gen_bus'):
        for i, bus in enumerate(lnd.gen_bus):
            bus_idx = bus_to_ind[bus]
            gen_output = abs(S_gen[bus_idx])
            limit = lnd.gen_MVA[i]
            if gen_output > limit:
                violations.append("Generator at bus {} violation: output {:.2f} MVA > limit {:.2f} MVA".format(
                    bus, gen_output, limit))
    
    # 3. Check bus voltage limits (must be between 0.9 and 1.1 pu)
    for i in range(len(V)):
        voltage = abs(V[i])
        if voltage < 0.9 or voltage > 1.1:
            violations.append("Voltage violation at bus {}: {:.2f} pu".format(
                lnd.ind_to_bus[i], voltage))
    
    return violations
