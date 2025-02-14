import numpy as np
import ReadNetworkData as rd


def load_network_data(filename):
    """
    Reads a network data file and constructs the network model.

    The function builds:
      - Bus admittance matrix (Ybus)
      - Branch admittance matrices (Y_fr and Y_to) for line flows
      - Arrays with branch indices (br_f, br_t)
      - Bus type codes and labels (buscode, bus_labels)
      - Complex load vector (S_LD in pu)
      - System MVA base and initial bus voltage vector (V0)
      - Optionally, indices for PQ, PV, and reference buses (pq_index, pv_index, ref)

    Parameters:
        filename (str): Path to the data file.

    Returns:
        tuple: (Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels,
                S_LD, MVA_base, V0, pq_index, pv_index, ref)
    """
    # Declare globals for use elsewhere if desired.
    global Ybus, Sbus, V0, buscode, ref, pq_index, pv_index
    global Y_fr, Y_to, br_f, br_t, S_LD, ind_to_bus, bus_to_ind
    global MVA_base, bus_labels, bus_kv, v_min, v_max

    # Read network data from file using the helper
    (bus_data, load_data, gen_data, line_data, tran_data, mva_base,
     bus_to_ind, ind_to_bus) = rd.read_network_data_from_file(filename)
    MVA_base = mva_base

    # Determine sizes for arrays
    num_buses = len(bus_data)
    num_lines = len(line_data)
    num_trans = len(tran_data)
    num_branches = num_lines + num_trans

    # Initialize matrices and arrays
    Ybus = np.zeros((num_buses, num_buses), dtype=complex)
    Y_fr = np.zeros((num_branches, num_buses), dtype=complex)
    Y_to = np.zeros((num_branches, num_buses), dtype=complex)
    br_f = np.zeros(num_branches, dtype=int)
    br_t = np.zeros(num_branches, dtype=int)

    branch_counter = 0
    branch_counter = _process_line_data(line_data, bus_to_ind, Ybus,
                                          Y_fr, Y_to, br_f, br_t,
                                          branch_counter)
    branch_counter = _process_transformer_data(tran_data, bus_to_ind, Ybus,
                                               Y_fr, Y_to, br_f, br_t,
                                               branch_counter)
    (buscode, bus_labels, V0, bus_kv, v_min, v_max) = _process_bus_data(
        bus_data, bus_to_ind
    )
    S_LD = _process_load_data(load_data, bus_to_ind, MVA_base, num_buses)
    pq_index, pv_index, ref = _classify_bus_types(buscode)

    # Optionally convert lists to numpy arrays
    pq_index = np.array(pq_index)
    pv_index = np.array(pv_index)

    return (Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels, S_LD,
            MVA_base, V0, pq_index, pv_index, ref)


def _process_line_data(line_data, bus_to_ind, Ybus, Y_fr, Y_to,
                       br_f, br_t, branch_counter):
    """
    Process line data (without tap ratios) to update Ybus and branch matrices.

    Expected line data format:
        [fr_bus, to_bus, ID, R, X, B, MVA_rat, X2, X0]
    """
    for ld in line_data:
        fr_bus, to_bus = ld[0], ld[1]
        R, X, B = ld[3], ld[4], ld[5]

        Z_series = complex(R, X)
        y_series = 1 / Z_series if Z_series != 0 else 0
        y_shunt = 1j * (B / 2)  # shunt susceptance split equally

        i = bus_to_ind[fr_bus]
        j = bus_to_ind[to_bus]

        # Update Ybus (series + shunt contributions)
        Ybus[i, i] += y_series + y_shunt
        Ybus[j, j] += y_series + y_shunt
        Ybus[i, j] -= y_series
        Ybus[j, i] -= y_series

        # Build branch admittance matrices
        Y_fr[branch_counter, i] = y_series + y_shunt
        Y_fr[branch_counter, j] = -y_series
        Y_to[branch_counter, i] = -y_series
        Y_to[branch_counter, j] = y_series + y_shunt

        # Save branch indices in bus-index space
        br_f[branch_counter] = i
        br_t[branch_counter] = j

        branch_counter += 1

    return branch_counter


def _process_transformer_data(tran_data, bus_to_ind, Ybus, Y_fr, Y_to,
                              br_f, br_t, branch_counter):
    """
    Process transformer data to update Ybus and branch matrices.

    Expected transformer data format:
        [fr_bus, to_bus, ID, R_eq, X_eq, n_pu, ang_deg,
         MVA_rat, fr_con, to_con, X2, X0]
    """
    for td in tran_data:
        fr_bus, to_bus = td[0], td[1]
        R_eq, X_eq = td[3], td[4]
        n_pu, ang_deg = td[5], td[6]

        Z_series = complex(R_eq, X_eq)
        y_series = 1 / Z_series if Z_series != 0 else 0
        y_shunt = 0  # No line charging for transformers

        # Calculate the complex tap ratio with phase shift
        a = n_pu * np.exp(1j * np.deg2rad(ang_deg))

        i = bus_to_ind[fr_bus]
        j = bus_to_ind[to_bus]

        # Update Ybus using the off-nominal tap model
        Ybus[i, i] += y_series / (abs(a) ** 2)
        Ybus[i, j] -= y_series / np.conjugate(a)
        Ybus[j, i] -= y_series / a
        Ybus[j, j] += y_series

        # Build branch admittance matrices for transformer
        Y_fr[branch_counter, i] = y_series / (abs(a) ** 2) + y_shunt
        Y_fr[branch_counter, j] = -y_series / np.conjugate(a)
        Y_to[branch_counter, i] = -y_series / a
        Y_to[branch_counter, j] = y_series + y_shunt

        br_f[branch_counter] = i
        br_t[branch_counter] = j

        branch_counter += 1

    return branch_counter


def _process_bus_data(bus_data, bus_to_ind):
    """
    Process bus data to extract bus codes, labels, initial voltage,
    and voltage limits.

    Expected bus data format:
        [bus_nr, label, v_init, theta_init, code, kv_level, v_low, v_high]
    """
    num_buses = len(bus_data)
    buscode = np.zeros(num_buses, dtype=int)
    bus_labels = [''] * num_buses
    V0 = np.zeros(num_buses, dtype=complex)
    bus_kv = np.zeros(num_buses)
    v_min = np.zeros(num_buses)
    v_max = np.zeros(num_buses)

    for b in bus_data:
        bus_nr = b[0]
        idx = bus_to_ind[bus_nr]
        bus_labels[idx] = b[1].strip() if isinstance(b[1], str) else str(b[1])
        v_init, theta_init = b[2], b[3]
        V0[idx] = v_init * np.exp(1j * np.deg2rad(theta_init))
        buscode[idx] = int(b[4])
        bus_kv[idx] = b[5]
        v_min[idx] = b[6]
        v_max[idx] = b[7]

    return buscode, bus_labels, V0, bus_kv, v_min, v_max


def _process_load_data(load_data, bus_to_ind, MVA_base, num_buses):
    """
    Process load data to create the complex load vector (S_LD) in per unit.

    Expected load data format:
        [bus_nr, P_LD_MW, Q_LD_MVAR]
    """
    S_LD = np.zeros(num_buses, dtype=complex)
    for ld in load_data:
        bus_nr = ld[0]
        idx = bus_to_ind[bus_nr]
        P_load_MW, Q_load_MVAR = ld[1], ld[2]
        S_LD[idx] = complex(P_load_MW, Q_load_MVAR) / MVA_base

    return S_LD


def _classify_bus_types(buscode):
    """
    Classify buses based on the bus code:
      - 1 indicates a PQ bus
      - 2 indicates a PV bus
      - 3 indicates the reference bus

    Returns:
        tuple: (pq_index, pv_index, ref)
    """
    pq_index = []
    pv_index = []
    ref = None

    for i, code in enumerate(buscode):
        if code == 1:
            pq_index.append(i)
        elif code == 2:
            pv_index.append(i)
        elif code == 3:
            ref = i

    return pq_index, pv_index, ref


# --- Testing block ---
if __name__ == '__main__':
    filename = 'testsystem.txt'
    (Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels, S_LD,
     MVA_base, V0, pq_index, pv_index, ref) = load_network_data(filename)

    # Formatter for printing floating-point and complex numbers
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
