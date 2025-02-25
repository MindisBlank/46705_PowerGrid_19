
def apply_contingency_to_Y_matrices(Ybus, Y_fr, Y_to, fr_ind, to_ind, br_ind, Ybr_mat):
    # Inputs:
    # Ybus  : Original bus admittance matrix
    # Y_fr  : Original branch-from admittance matrix (for line flows)
    # Y_to  : Original branch-to admittance matrix (for line flows)
    # fr_ind: From-bus index of the branch
    # to_ind: To-bus index of the branch
    # br_ind: Index of the branch in the branch list
    # Ybr_mat: 2x2 admittance matrix of the branch to be removed

    # Copy the original matrices to avoid altering them
    Ybus_mod = Ybus.copy()
    Y_fr_mod = Y_fr.copy()
    Y_to_mod = Y_to.copy()

    # 1. Remove the branch from the bus admittance matrix:
    # Subtract the branch admittance contributions from the appropriate bus entries.
    Ybus_mod[fr_ind, fr_ind] -= Ybr_mat[0, 0]
    Ybus_mod[to_ind, to_ind] -= Ybr_mat[1, 1]
    Ybus_mod[fr_ind, to_ind] -= Ybr_mat[0, 1]
    Ybus_mod[to_ind, fr_ind] -= Ybr_mat[1, 0]

    # 2. Remove the branch from the branch flow matrices:
    # Zero out the row corresponding to the branch in both Y_fr and Y_to.
    Y_fr_mod[br_ind, :] = 0
    Y_to_mod[br_ind, :] = 0

    return Ybus_mod, Y_fr_mod, Y_to_mod






def System_violations(V, Ybus, Y_from, Y_to, lnd):
    # store variables as more convenient names
    br_f = lnd.br_f              # array of from-bus indices (0-indexed)
    br_t = lnd.br_t              # array of to-bus indices (0-indexed)
    BUS_NR = lnd.BUS_NR          # list of bus numbers for identification
    branch_info = lnd.FROM_BUS_AND_TO_BUS  # list of (from bus, to bus, branch id)
    
    # Calculate line flows and injections
    S_to = V[br_t] * (Y_to.dot(V)).conj()   # flow into the 'to' end of each branch
    S_from = V[br_f] * (Y_from.dot(V)).conj() # flow into the 'from' end of each branch
    S_inj = V * (Ybus.dot(V)).conj()          # net injection at each bus
    SLD = lnd.S_LD                          # defined loads on PQ buses
    S_gen = S_inj + SLD                      # generator outputs (net injections plus loads)

    violations = []  # empty list to store descriptions of any violations

    # 1. Check branch flows for violations
    num_branches = len(br_f)
    num_lines = len(lnd.Br_rating)  # number of branches from line data
    for i in range(num_branches):
        #print(lnd.Br_rating[i][3])
        # Determine branch rating: for the first num_lines branches, use Br_rating;
        # for remaining branches, use Tran_rating.
        if i < num_lines:
            limit = lnd.Br_rating[i][3]  # tuple structure: (from bus, to bus, ID, MVA rating)
        else:
            limit = lnd.Tran_rating[i - num_lines][3]
        flow_from = abs(S_from[i])*lnd.MVA_base
        flow_to = abs(S_to[i])*lnd.MVA_base
        branch_id = branch_info[i][2]
        from_bus = branch_info[i][0]
        to_bus = branch_info[i][1]
        if flow_from > limit:
            violations.append("Branch {} (bus {} -> bus {}) from-end flow violation: {:.2f} MVA > limit {:.2f} MVA".format(
                branch_id, from_bus, to_bus, flow_from, limit))
        if flow_to > limit:
            violations.append("Branch {} (bus {} -> bus {}) to-end flow violation: {:.2f} MVA > limit {:.2f} MVA".format(
                branch_id, from_bus, to_bus, flow_to, limit))
    
    # 2. Check generator outputs for violations
    # Each element in Gen_rating is a tuple: (bus number, MVA rating)
    for (gen_bus, gen_limit) in lnd.Gen_rating:
        try:
            bus_idx = BUS_NR.index(gen_bus)
        except ValueError:
            continue  # skip if the generator bus is not found
        gen_output  = abs(S_gen[bus_idx])* lnd.MVA_base
        if gen_output  > gen_limit:
            violations.append("Generator at bus {} violation: output {:.2f} MVA > limit {:.2f} MVA".format(
                gen_bus, gen_output , gen_limit))
    
    # 3. Check bus voltages (assumed limits: 0.9 pu to 1.1 pu)
    for i in range(len(V)):
        voltage = abs(V[i])
        print(voltage)
        if voltage < lnd.v_min[i] or voltage > lnd.v_max[i]:
            violations.append("Voltage violation at bus {}: {:.2f} pu".format(
                BUS_NR[i], voltage))
    
    return violations


