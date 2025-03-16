
def apply_contingency_to_Y_matrices(ybus, y_fr, y_to, fr_ind, to_ind, br_ind, ybr_mat):
    """
    Apply contingency to Y matrices.

    Parameters:
        ybus (ndarray): Original bus admittance matrix.
        y_fr (ndarray): Original branch-from admittance matrix (for line flows).
        y_to (ndarray): Original branch-to admittance matrix (for line flows).
        fr_ind (int): From-bus index of the branch.
        to_ind (int): To-bus index of the branch.
        br_ind (int): Index of the branch in the branch list.
        ybr_mat (ndarray): 2x2 admittance matrix of the branch to be removed.

    Returns:
        tuple: Modified ybus, y_fr, and y_to matrices.
    """
    # Copy the original matrices to avoid altering them
    ybus_mod = ybus.copy()
    y_fr_mod = y_fr.copy()
    y_to_mod = y_to.copy()

    # 1. Remove the branch from the bus admittance matrix:
    # Subtract the branch admittance contributions from the appropriate bus entries.
    ybus_mod[fr_ind, fr_ind] -= ybr_mat[0, 0]
    ybus_mod[to_ind, to_ind] -= ybr_mat[1, 1]
    ybus_mod[fr_ind, to_ind] -= ybr_mat[0, 1]
    ybus_mod[to_ind, fr_ind] -= ybr_mat[1, 0]

    # 2. Remove the branch from the branch flow matrices:
    # Zero out the row corresponding to the branch in both y_fr and y_to.
    y_fr_mod[br_ind, :] = 0
    y_to_mod[br_ind, :] = 0

    return ybus_mod, y_fr_mod, y_to_mod

#TODO fix branch ID numbers
def system_violations(v, ybus, y_from, y_to, lnd):
    """
    Calculate system violations based on voltage, branch flows, and generator outputs.

    Parameters:
        v (ndarray): Voltage vector.
        ybus (ndarray): Bus admittance matrix.
        y_from (ndarray): Branch-from admittance matrix.
        y_to (ndarray): Branch-to admittance matrix.
        lnd (object): Network data object containing attributes such as:
            - branch_from: Array of from-bus indices (0-indexed).
            - branch_to: Array of to-bus indices (0-indexed).
            - bus_numbers: List of bus numbers for identification.
            - bus_pairs: List of (from bus, to bus, branch id) tuples.
            - S_load: Defined loads on PQ buses.
            - MVA_base: System base in MVA.
            - branch_rating: List of branch ratings from line data.
            - tran_rating: List of transformer ratings.
            - gen_rating: List of (generator bus, MVA rating, Q_max [MVAr], Q_min [MVAr]) tuples.
            - v_min: Minimum voltage limits.
            - v_max: Maximum voltage limits.

    Returns:
        list: A list of violation description strings.
    """
    # Store variables with more convenient names
    br_f = lnd.branch_from         # Array of from-bus indices (0-indexed)
    br_t = lnd.branch_to           # Array of to-bus indices (0-indexed)
    bus_nr = lnd.bus_numbers       # List of bus numbers for identification
    branch_info = lnd.bus_pairs    # List of (from bus, to bus, branch id)

    # Calculate line flows and injections
    s_to = v[br_t] * (y_to.dot(v)).conj()     # Flow into the 'to' end of each branch
    s_from = v[br_f] * (y_from.dot(v)).conj()   # Flow into the 'from' end of each branch
    s_inj = v * (ybus.dot(v)).conj()            # Net injection at each bus
    s_ld = lnd.S_load                         # Defined loads on PQ buses
    s_gen = s_inj + s_ld                      # Generator outputs (net injections plus loads)

    violations = []  # List to store descriptions of any violations

    # 1. Check branch flows for violations
    num_branches = len(br_f)
    num_lines = len(lnd.branch_rating)  # Number of branches from line data
    for i in range(num_branches):
        # Determine branch rating: for the first num_lines branches, use branch_rating;
        # for remaining branches, use tran_rating.
        if i < num_lines:
            limit = lnd.branch_rating[i][3]  # Tuple structure: (from bus, to bus, ID, MVA rating)
        else:
            limit = lnd.tran_rating[i - num_lines][3]

        flow_from = abs(s_from[i]) * lnd.MVA_base
        flow_to = abs(s_to[i]) * lnd.MVA_base
        branch_num = i+1  # Branch ID number (1-indexed)
        from_bus = branch_info[i][0]
        to_bus = branch_info[i][1]

        if flow_from > limit:
            violations.append(
                "Branch {} (bus {} -> bus {}) from-end flow violation: "
                "{:.2f} MVA > limit {:.2f} MVA".format(
                    branch_num, from_bus, to_bus, flow_from, limit
                )
            )
        if flow_to > limit:
            violations.append(
                "Branch {} (bus {} -> bus {}) to-end flow violation: "
                "{:.2f} MVA > limit {:.2f} MVA".format(
                    branch_num, from_bus, to_bus, flow_to, limit
                )
            )

    # 2. Check generator outputs for violations (including reactive power limits)
    for gen_bus, mva_rating, Q_max, Q_min in lnd.gen_rating:
        try:
            bus_idx = bus_nr.index(gen_bus)
        except ValueError:
            continue  # Skip if the generator bus is not found

        # Calculate generator injection in MVA (complex number)
        gen_injection = s_gen[bus_idx] * lnd.MVA_base
        apparent_power = abs(gen_injection)
        if apparent_power > mva_rating:
            violations.append(
                "Generator at bus {} violation: Apparent power {:.2f} MVA > rating {:.2f} MVA".format(
                    gen_bus, apparent_power, mva_rating
                )
            )

        # Check reactive power limits
        Q_gen = gen_injection.imag
        if Q_gen > Q_max:
            violations.append(
                "Generator at bus {} reactive power violation: Q {:.2f} MVAr > Q_max {:.2f} MVAr".format(
                    gen_bus, Q_gen, Q_max
                )
            )
        if Q_gen < Q_min:
            violations.append(
                "Generator at bus {} reactive power violation: Q {:.2f} MVAr < Q_min {:.2f} MVAr".format(
                    gen_bus, Q_gen, Q_min
                )
            )

    # 3. Check voltage limits for violations
    for i in range(len(v)):
        voltage = abs(v[i])
        if voltage < lnd.v_min[i] or voltage > lnd.v_max[i]:
            violations.append(
                "Voltage violation at bus {}: {:.2f} pu".format(bus_nr[i], voltage)
            )

    return violations
