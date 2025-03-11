import numpy as np
import ReadNetworkData as rd
from logger import log_function, setup_logger
from dataclasses import dataclass

setup_logger()
@dataclass
class NetworkData:
    Ybus: np.ndarray
    Y_from: np.ndarray
    Y_to: np.ndarray
    branch_from: np.ndarray
    branch_to: np.ndarray
    buscode: list
    bus_labels: list
    Sbus: np.ndarray
    S_load: np.ndarray
    MVA_base: float
    V0: np.ndarray
    pq_index: np.ndarray
    pv_index: np.ndarray
    ref: list
    gen_rating: list
    branch_rating: list
    bus_numbers: list
    bus_pairs: list
    tran_rating: list
    v_min: float
    v_max: float

@log_function
def load_network_data(filename: str, debug: bool = False) -> NetworkData:
    # Read network data from file using a helper function
    bus_data, load_data, gen_data, line_data, tran_data, mva_base, bus_to_ind, ind_to_bus = rd.read_network_data_from_file(filename)
    
    # Extract ratings and bus numbers 
    gen_rating = [(gen[0], gen[1],gen[4],gen[5]) for gen in gen_data]
    branch_rating = [(br[0], br[1], br[2], br[6]) for br in line_data]
    tran_rating = [(tran[0], tran[1], tran[2], tran[7]) for tran in tran_data]
    bus_numbers = [bus[0] for bus in bus_data]
    bus_pairs = [(line[0], line[1], line[2]) for line in line_data] + [(tran[0], tran[1], tran[2]) for tran in tran_data]

    num_buses = len(bus_data)
    num_lines = len(line_data)
    num_trans = len(tran_data)
    num_branches = num_lines + num_trans

    # Initialize matrices
    Ybus = np.zeros((num_buses, num_buses), dtype=complex)
    Y_from = np.zeros((num_branches, num_buses), dtype=complex)
    Y_to = np.zeros((num_branches, num_buses), dtype=complex)
    branch_from = np.zeros(num_branches, dtype=int)
    branch_to = np.zeros(num_branches, dtype=int)

    branch_counter = 0
    branch_counter = _process_line_data(line_data, bus_to_ind, Ybus, Y_from, Y_to, branch_from, branch_to, branch_counter)
    branch_counter = _process_transformer_data(tran_data, bus_to_ind, Ybus, Y_from, Y_to, branch_from, branch_to, branch_counter)
    
    buscode, bus_labels, V0, bus_kv, v_min, v_max = _process_bus_data(bus_data, bus_to_ind)
    S_load = _process_load_data(load_data, bus_to_ind, mva_base, num_buses)
    S_gen = _process_gen_data(gen_data, bus_to_ind, mva_base, num_buses)

    # Compute Sbus using a list comprehension
    Sbus = np.array([0 if code == 3 else S_gen[i] - S_load[i] for i, code in enumerate(buscode)], dtype=complex)
    pq_index, pv_index, ref = _classify_bus_types(buscode)
    
    return NetworkData(
        Ybus=Ybus,
        Y_from=Y_from,
        Y_to=Y_to,
        branch_from=branch_from,
        branch_to=branch_to,
        buscode=buscode,
        bus_labels=bus_labels,
        Sbus=Sbus,
        S_load=S_load,
        MVA_base=mva_base,
        V0=V0,
        pq_index=np.array(pq_index),
        pv_index=np.array(pv_index),
        ref=ref,
        gen_rating=gen_rating,
        branch_rating=branch_rating,
        bus_numbers=bus_numbers,
        bus_pairs=bus_pairs,
        tran_rating=tran_rating,
        v_min=v_min,
        v_max=v_max
    )

def _process_line_data(line_data, bus_to_ind, Ybus, Y_fr, Y_to, br_f, br_t, branch_counter):
    for ld in line_data:
        fr_bus, to_bus = ld[0], ld[1]
        R, X, B = ld[3], ld[4], ld[5]

        Z_series = complex(R, X)
        y_series = 1 / Z_series if Z_series != 0 else 0
        y_shunt = 1j * (B / 2)  # shunt susceptance split equally

        i = bus_to_ind[fr_bus]
        j = bus_to_ind[to_bus]

        Ybus[i, i] += y_series + y_shunt
        Ybus[j, j] += y_series + y_shunt
        Ybus[i, j] -= y_series
        Ybus[j, i] -= y_series

        Y_fr[branch_counter, i] = y_series + y_shunt
        Y_fr[branch_counter, j] = -y_series
        Y_to[branch_counter, i] = -y_series
        Y_to[branch_counter, j] = y_series + y_shunt

        br_f[branch_counter] = i
        br_t[branch_counter] = j

        branch_counter += 1

    return branch_counter


def _process_transformer_data(tran_data, bus_to_ind, Ybus, Y_fr, Y_to, br_f, br_t, branch_counter):
    for td in tran_data:
        fr_bus, to_bus = td[0], td[1]
        R_eq, X_eq = td[3], td[4]
        n_pu, ang_deg = td[5], td[6]

        Z_series = complex(R_eq, X_eq)
        y_series = 1 / Z_series if Z_series != 0 else 0
        y_shunt = 0  # No line charging for transformers

        a = n_pu * np.exp(1j * np.deg2rad(ang_deg))

        i = bus_to_ind[fr_bus]
        j = bus_to_ind[to_bus]

        Ybus[i, i] += y_series / (abs(a) ** 2)
        Ybus[i, j] -= y_series / np.conjugate(a)
        Ybus[j, i] -= y_series / a
        Ybus[j, j] += y_series

        Y_fr[branch_counter, i] = y_series / (abs(a) ** 2) + y_shunt
        Y_fr[branch_counter, j] = -y_series / np.conjugate(a)
        Y_to[branch_counter, i] = -y_series / a
        Y_to[branch_counter, j] = y_series + y_shunt

        br_f[branch_counter] = i
        br_t[branch_counter] = j

        branch_counter += 1

    return branch_counter


def _process_bus_data(bus_data, bus_to_ind):
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
    S_LD = np.zeros(num_buses, dtype=complex)
    for ld in load_data:
        bus_nr = ld[0]
        idx = bus_to_ind[bus_nr]
        P_load_MW, Q_load_MVAR = ld[1], ld[2]
        S_LD[idx] = complex(P_load_MW, Q_load_MVAR) / MVA_base

    return S_LD


def _process_gen_data(gen_data, bus_to_ind, MVA_base, num_buses):
    """
    Process generator data to form the S_gen vector.
    Expected generator data format:
        [BUS_NR, MVA_SIZE, P_GEN [MW], P_max [MW], Q_max [MVAr], Q_min [MVAr], ...]
    We use the P_GEN value (in MW) and assume Q_GEN = 0 for PV buses.
    """
    S_gen = np.zeros(num_buses, dtype=complex)
    for gd in gen_data:
        bus_nr = gd[0]
        idx = bus_to_ind[bus_nr]
        P_gen = gd[2]  # P_GEN in MW
        # For simplicity, assume Q_gen = 0 (PV buses)
        S_gen[idx] += complex(P_gen / MVA_base, 0)
    return S_gen


def _classify_bus_types(buscode):
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


