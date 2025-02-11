import numpy as np
import ReadNetworkData as rd

def LoadNetworkData(filename):
    """
    Reads a network data file and constructs:
      - Bus admittance matrix Ybus
      - Branch admittance matrices Y_fr and Y_to for calculating line flows
      - Arrays with branch indices: br_f (from bus indices), br_t (to bus indices)
      - Bus type codes and labels (buscode, bus_labels)
      - Complex load vector S_LD (in pu, on system base)
      - The system MVA base, and the initial bus voltage vector V0
      - Optionally, indices for PQ, PV and reference buses (pq_index, pv_index, ref)
    
    Input:
      filename : name (or path) of the data file.
    
    Returns (and sets as globals):
      Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels, S_LD, MVA_base, V0,
      pq_index, pv_index, ref
    """
    # Declare globals (if you want these available throughout your program)
    global Ybus, Sbus, V0, buscode, ref, pq_index, pv_index, Y_fr, Y_to, br_f, br_t
    global S_LD, ind_to_bus, bus_to_ind, MVA_base, bus_labels, bus_kv, v_min, v_max

    # Read in the network data (the helper returns the various data lists)
    bus_data, load_data, gen_data, line_data, tran_data, mva_base, bus_to_ind, ind_to_bus = \
            rd.read_network_data_from_file(filename)

    # Set system MVA base
    MVA_base = mva_base

    # Number of buses and branches
    N = len(bus_data)            # number of buses
    M_lines = len(line_data)     # number of lines
    M_trans = len(tran_data)     # number of transformers
    M_branches = M_lines + M_trans  # total number of branches

    # Initialize bus admittance matrix (N x N)
    Ybus = np.zeros((N, N), dtype=complex)

    # Initialize branch admittance matrices (each of size [M_branches x N])
    Y_fr = np.zeros((M_branches, N), dtype=complex)
    Y_to = np.zeros((M_branches, N), dtype=complex)

    # Arrays that keep the bus indices (as given in bus_to_ind) for each branch’s ends
    br_f = np.zeros(M_branches, dtype=int)
    br_t = np.zeros(M_branches, dtype=int)

    # A branch counter helps us fill the branch-related matrices in order.
    branch_counter = 0

    # ===============================
    # Process line data (lines have no tap ratio)
    # ===============================
    for ld in line_data:
        # Expected structure: [fr_bus, to_bus, ID, R, X, B, MVA_rat, X2, X0]
        fr_bus = ld[0]
        to_bus = ld[1]
        R = ld[3]
        X = ld[4]
        B = ld[5]

        # Compute series impedance and its reciprocal (series admittance)
        Z_series = complex(R, X)
        y_series = 1 / Z_series if Z_series != 0 else 0

        # For lines the shunt susceptance B (in pu) is usually split equally at both ends.
        y_shunt = 1j * (B / 2)

        # Map bus numbers to indices (using the provided mapping)
        i = bus_to_ind[fr_bus]
        j = bus_to_ind[to_bus]

        # --- Update the bus admittance matrix Ybus ---
        # Add series and shunt admittance at both ends.
        Ybus[i, i] += y_series + y_shunt
        Ybus[j, j] += y_series + y_shunt
        Ybus[i, j] -= y_series
        Ybus[j, i] -= y_series

        # --- Build branch admittance matrices ---
        # For the "from" end:
        Y_fr[branch_counter, i] = y_series + y_shunt
        Y_fr[branch_counter, j] = -y_series
        # For the "to" end:
        Y_to[branch_counter, i] = -y_series
        Y_to[branch_counter, j] = y_series + y_shunt

        # Save branch “from” and “to” bus indices (in bus-index space)
        br_f[branch_counter] = i
        br_t[branch_counter] = j

        branch_counter += 1

    # ===============================
    # Process transformer data
    # ===============================
    for td in tran_data:
        # Expected structure: [fr_bus, to_bus, ID, R_eq, X_eq, n_pu, ang_deg, MVA_rat, fr_con, to_con, X2, X0]
        fr_bus = td[0]
        to_bus = td[1]
        R_eq = td[3]
        X_eq = td[4]
        n_pu = td[5]
        ang_deg = td[6]

        # Series impedance and its reciprocal (series admittance)
        Z_series = complex(R_eq, X_eq)
        y_series = 1 / Z_series if Z_series != 0 else 0

        # For this transformer model, we assume no line charging so shunt = 0.
        y_shunt = 0

        # Compute the complex tap ratio (magnitude and phase shift)
        a = n_pu * np.exp(1j * np.deg2rad(ang_deg))
        
        #n_pu= vindingar á eingur   

        # Map bus numbers to indices
        i = bus_to_ind[fr_bus]
        j = bus_to_ind[to_bus]

        # --- Update Ybus for transformer ---
        # A common model for off-nominal tap transformers is:
        #   Ybus[i,i] += y_series / |a|^2
        #   Ybus[i,j] -= y_series / conj(a)
        #   Ybus[j,i] -= y_series / a
        #   Ybus[j,j] += y_series
        Ybus[i, i] += y_series / (abs(a) ** 2)
        Ybus[i, j] -= y_series / np.conjugate(a)
        Ybus[j, i] -= y_series / a
        Ybus[j, j] += y_series

        # --- Build branch admittance matrices for transformer ---
        # In a similar spirit as for lines:
        Y_fr[branch_counter, i] = y_series / (abs(a) ** 2) + y_shunt  # shunt is zero here
        Y_fr[branch_counter, j] = - y_series / np.conjugate(a)
        Y_to[branch_counter, i] = - y_series / a
        Y_to[branch_counter, j] = y_series + y_shunt

        # Save branch indices
        br_f[branch_counter] = i
        br_t[branch_counter] = j

        branch_counter += 1

    # ===============================
    # Process bus data: assign bus codes, labels and initial voltages
    # ===============================
    buscode = np.zeros(N, dtype=int)
    bus_labels = [''] * N
    V0 = np.zeros(N, dtype=complex)
    bus_kv = np.zeros(N)
    v_min = np.zeros(N)
    v_max = np.zeros(N)

    # bus_data elements: [bus_nr, label, v_init, theta_init, code, kv_level, v_low, v_high]
    for b in bus_data:
        bus_nr = b[0]
        idx = bus_to_ind[bus_nr]
        bus_labels[idx] = b[1].strip() if isinstance(b[1], str) else str(b[1])
        v_init = b[2]
        theta_init = b[3]
        # Represent the initial voltage as a complex number (magnitude and angle in radians)
        V0[idx] = v_init * np.exp(1j * np.deg2rad(theta_init))
        buscode[idx] = int(b[4])
        bus_kv[idx] = b[5]
        v_min[idx] = b[6]
        v_max[idx] = b[7]

    # ===============================
    # Process load data: create complex load vector S_LD (in pu)
    # ===============================
    S_LD = np.zeros(N, dtype=complex)
    # load_data elements: [bus_nr, P_LD_MW, Q_LD_MVAR]
    for ld in load_data:
        bus_nr = ld[0]
        idx = bus_to_ind[bus_nr]
        P_load_MW = ld[1]
        Q_load_MVAR = ld[2]
        # Convert from MW and MVAr to per-unit (pu) using the system MVA base.
        S_LD[idx] = complex(P_load_MW, Q_load_MVAR) / MVA_base

    # ===============================
    # Identify bus types (PQ, PV, reference) for later use in power flow
    # ===============================
    pq_index = []  # PQ bus indices
    pv_index = []  # PV bus indices
    ref = None     # Reference bus index
    for i in range(N):
        if buscode[i] == 1:      # PQ bus
            pq_index.append(i)
        elif buscode[i] == 2:    # PV bus
            pv_index.append(i)
        elif buscode[i] == 3:    # Reference bus
            ref = i

    # Optionally convert indices to numpy arrays
    pq_index = np.array(pq_index)
    pv_index = np.array(pv_index)

    # (Other data such as generator data can be processed here as needed.)

    # For convenience, you may return the variables (or rely on globals)
    return (Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels, S_LD,
            MVA_base, V0, pq_index, pv_index, ref)






# --- For testing purposes ---
if __name__ == '__main__':
    filename = 'testsystem.txt'
    (Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels, S_LD,
     MVA_base, V0, pq_index, pv_index, ref) = LoadNetworkData(filename)

    # Define a formatter dictionary for floating-point and complex numbers.
formatter = {
    'float_kind': lambda x: f"{x:.3f}",
    'complex_kind': lambda x: f"{x.real:.3f}{'+' if x.imag >= 0 else ''}{x.imag:.3f}j"
}

print("Ybus =\n", np.array2string(Ybus, formatter=formatter))
print("\nY_fr =\n", np.array2string(Y_fr, formatter=formatter))
print("\nY_to =\n", np.array2string(Y_to, formatter=formatter))
print("\nBranch from indices:", br_f)
print("Branch to indices:", br_t)
print("\nBus codes:", buscode)
print("Bus labels:", bus_labels)
print("\nLoad vector S_LD (pu):", np.array2string(S_LD, formatter=formatter))
print("MVA Base:", f"{MVA_base:.3f}")
print("Initial Voltages V0:", np.array2string(V0, formatter=formatter))
print("PQ bus indices:", pq_index)
print("PV bus indices:", pv_index)
print("Reference bus index:", ref)
