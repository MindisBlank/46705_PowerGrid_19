import numpy as np
import ReadNetworkData as rd

def LoadNetworkData(filename):
    global Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels, S_LD, MVA_base

    # Read in the data from the file using ReadNetworkData
    bus_data, load_data, gen_data, line_data, tran_data, mva_base, bus_to_ind, ind_to_bus = \
        rd.read_network_data_from_file(filename)

    # System MVA base
    MVA_base = mva_base

    # Number of buses and branches in the system
    N = len(bus_data)  # Number of buses
    M_lines = len(line_data)
    M_trans = len(tran_data)
    M_branches = M_lines + M_trans

    # Initialize matrices and arrays
    Ybus = np.zeros((N, N), dtype=complex)  # Bus admittance matrix
    Y_fr = np.zeros((M_branches, N), dtype=complex)  # From-bus admittance matrix
    Y_to = np.zeros((M_branches, N), dtype=complex)  # To-bus admittance matrix
    br_f = np.zeros(M_branches, dtype=int)  # From-bus indices for branches
    br_t = np.zeros(M_branches, dtype=int)  # To-bus indices for branches
    buscode = np.zeros(N, dtype=int)  # Bus type codes (0 = PQ, 1 = PV, 2 = Slack)
    bus_labels = []  # Text labels for each bus
    S_LD = np.zeros(N, dtype=complex)  # Complex power load at each bus

    # Populate bus_labels and buscode
    for i, bus in enumerate(bus_data):
        bus_labels.append(bus[1])  # bus[1] is the label
        buscode[i] = bus[4]  # bus[4] is the buscode (0 = PQ, 1 = PV, 2 = Slack)

    # Populate S_LD (complex power load at each bus)
    for load in load_data:
        bus_idx = bus_to_ind[load[0]]  # load[0] is the bus number
        S_LD[bus_idx] = load[1] + 1j * load[2]  # load[1] is P, load[2] is Q

    # Populate Ybus, Y_fr, Y_to, br_f, and br_t
    branch_idx = 0

    # Process lines
    for line in line_data:
        from_bus = bus_to_ind[line[0]]  # line[0] is the from_bus
        to_bus = bus_to_ind[line[1]]  # line[1] is the to_bus
        R = line[3]  # line[3] is R
        X = line[4]  # line[4] is X
        B = line[5]  # line[5] is B (shunt susceptance)

        Y_series = 1 / (R + 1j * X)  # Series admittance
        Y_shunt = 1j * B / 2  # Shunt admittance (half at each end)

        # Update Ybus
        Ybus[from_bus, from_bus] += Y_series + Y_shunt
        Ybus[to_bus, to_bus] += Y_series + Y_shunt
        Ybus[from_bus, to_bus] -= Y_series
        Ybus[to_bus, from_bus] -= Y_series

        # Update Y_fr and Y_to
        Y_fr[branch_idx, from_bus] = Y_series + Y_shunt
        Y_fr[branch_idx, to_bus] = -Y_series
        Y_to[branch_idx, from_bus] = -Y_series
        Y_to[branch_idx, to_bus] = Y_series + Y_shunt

        # Update branch indices
        br_f[branch_idx] = from_bus
        br_t[branch_idx] = to_bus

        branch_idx += 1

    # Process transformers
    for tran in tran_data:
        from_bus = bus_to_ind[tran[0]]  # tran[0] is the from_bus
        to_bus = bus_to_ind[tran[1]]  # tran[1] is the to_bus
        R = tran[3]  # tran[3] is R
        X = tran[4]  # tran[4] is X
        n = tran[5]  # tran[5] is the turns ratio
        ang1 = tran[6]  # tran[6] is the phase shift angle

        Y_tran = 1 / (R + 1j * X)  # Transformer admittance

        # Update Ybus
        Ybus[from_bus, from_bus] += Y_tran
        Ybus[to_bus, to_bus] += Y_tran
        Ybus[from_bus, to_bus] -= Y_tran
        Ybus[to_bus, from_bus] -= Y_tran

        # Update Y_fr and Y_to
        Y_fr[branch_idx, from_bus] = Y_tran
        Y_fr[branch_idx, to_bus] = -Y_tran
        Y_to[branch_idx, from_bus] = -Y_tran
        Y_to[branch_idx, to_bus] = Y_tran

        # Update branch indices
        br_f[branch_idx] = from_bus
        br_t[branch_idx] = to_bus

        branch_idx += 1

    # Return the created matrices and arrays
    print(len(tran_data))
    return Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels, S_LD, MVA_base

# Example usage
LoadNetworkData('TestSystem.txt')