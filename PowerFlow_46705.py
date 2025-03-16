"""
46705 - Power Grid Analysis
This file contains the definitions of the functions needed to
carry out Power Flow calculations in python.

How to carry out Power Flow in a new *.py file? 
See the example in table 1 in the assignment text
"""
import numpy as np
from logger import log_function, setup_logger

setup_logger()


@log_function
def PowerFlowNewton(Ybus, Sbus, V0, pv_index, pq_index, max_iter, err_tol, print_progress=True, debug=False):
    """
    Solve the power flow equations using the Newton-Raphson method.
    Optionally logs detailed debug information if debug is True.
    
    Parameters:
        Ybus (ndarray): N x N bus admittance matrix.
        Sbus (ndarray): N x 1 specified complex power injection vector (in pu).
        V0 (ndarray): Initial bus voltage vector (complex values).
        pv_index (list or ndarray): Indices of PV buses.
        pq_index (list or ndarray): Indices of PQ buses.
        max_iter (int): Maximum number of iterations.
        err_tol (float): Tolerance for convergence.
        print_progress (bool): Flag to print progress information.
        debug (bool): Flag to enable detailed logging.
    
    Returns:
        tuple: (V, success, n) where:
            V (ndarray): Final bus voltage vector (complex).
            success (int): 1 if the solution converged, 0 otherwise.
            n (int): Number of iterations used.
    """
    # Initialization of the status flag and iteration counter
    success = 0
    n = 0
    V = V0
    
    if print_progress:
        print(' iteration maximum P & Q mismatch (pu)')
        print(' --------- ---------------------------')
    
    # Determine mismatch between initial guess and specified power injections
    
    F = calculate_F(Ybus, Sbus, V, pv_index, pq_index)

    # Check if the desired tolerance is reached
    success = CheckTolerance(F, n, err_tol, print_progress)
    # Start the Newton-Raphson iteration loop
    while (not success) and (n < max_iter):
        n += 1
        J_dS_dVm, J_dS_dTheta = generate_Derivatives(Ybus, V)
        J = generate_Jacobian(J_dS_dVm, J_dS_dTheta, pv_index, pq_index)
        dx = np.linalg.solve(J, F)
        V = Update_Voltages(dx, V, pv_index, pq_index)
        F = calculate_F(Ybus, Sbus, V, pv_index, pq_index)
        success = CheckTolerance(F, n, err_tol, print_progress)
    # Display convergence message if required
    if success:
        if print_progress:
            print('The Newton-Raphson Power Flow Converged in %d iterations!' % n)
    else:
        if print_progress:
            print('No Convergence !!!\nStopped after %d iterations without solution...' % n)
    return V, success, n


def calculate_F(Ybus,Sbus,V,pv_index,pq_index):
    """
    Calculate the power mismatch vector F.

    The mismatch is defined as the difference between the specified and calculated 
    power injections. For bus i, the calculated complex power is:
        S_calc,i = V_i * conj( sum_j (Ybus[i,j] * V_j) )
    Then:
        ΔP_i = P_spec,i - Re(S_calc,i)
        ΔQ_i = Q_spec,i - Im(S_calc,i)
    
    For PV buses, only the active power mismatch is used (ΔP).
    For PQ buses, both active and reactive mismatches (ΔP and ΔQ) are considered.
    
    Parameters:
        Ybus (ndarray): N x N bus admittance matrix.
        Sbus (ndarray): N x 1 complex vector of specified power injections (in pu).
        V (ndarray): N x 1 complex vector of bus voltages.
        pv_index (ndarray or list): Indices of PV buses.
        pq_index (ndarray or list): Indices of PQ buses.
    
    Returns:
        F (ndarray): Mismatch vector. The ordering is:
            [ ΔP for all PV buses and PQ buses, ΔQ for all PQ buses ]
    """
    # Calculate the injected complex power based on current voltage estimate
    bus_current = Ybus @ V
    S_calc = V * np.conjugate(bus_current)
    
    # Separate specified and calculated active and reactive powers
    P_spec = Sbus.real
    Q_spec = Sbus.imag
    P_calc = S_calc.real
    Q_calc = S_calc.imag

    # Calculate mismatches
    dP = P_spec - P_calc  # active power mismatch for all buses
    dQ = Q_spec - Q_calc  # reactive power mismatch for all buses

    # Assemble mismatch vector:
    # - For all PV buses and PQ buses, use the active power mismatch.
    # - For PQ buses, use the reactive power mismatch.
    # Order: [ΔP(pv) ; ΔP(pq) ; ΔQ(pq)]
    indices_active = np.concatenate((pv_index, pq_index))
    F_P = dP[indices_active]
    F_Q = dQ[pq_index]
    
    # Concatenate mismatches into one vector F
    F = np.concatenate((F_P, F_Q))
    return F


def CheckTolerance(F, n, err_tol, print_progress=True):
    """
    Check whether the current mismatch vector F is within the error tolerance.

    Parameters:
        F (ndarray): Mismatch vector from the power flow equations.
        n (int): Current iteration number.
        err_tol (float): Error tolerance threshold.
        print_progress (bool): Flag indicating whether to print progress.

    Returns:
        success (int): 1 if the maximum mismatch is less than err_tol, otherwise 0.
    """
    # Compute the maximum absolute mismatch
    max_mismatch = np.linalg.norm(F,np.inf)
    # Optionally print progress information
    if print_progress:
        print(f"Iteration {n}: maximum mismatch = {max_mismatch:.6f} (tolerance = {err_tol})")
    
    # If the maximum mismatch is below the tolerance, consider the solution converged
    success = 1 if max_mismatch < err_tol else 0
    
    return success


def generate_Derivatives(Ybus, V):
    """
    Calculates the derivatives of the complex power S = V · conj(I) with respect
    to the voltage magnitude and voltage angle.
    
    Using the formulas:
      dS/dV = diag(V/|V|) · (diag(Ybus.dot(V)))∗ + diag(V) · (Ybus.dot(diag(V/|V|)))∗
      dS/dθ = j · diag(V) · (diag(Ybus.dot(V)) - Ybus.dot(diag(V)))∗
    
    Parameters:
        Ybus (ndarray): N×N bus admittance matrix.
        V (ndarray): N×1 complex voltage vector.
    
    Returns:
        J_dS_dVm (ndarray): N×N complex matrix of derivatives of S with respect to voltage magnitude.
        J_dS_dTheta (ndarray): N×N complex matrix of derivatives of S with respect to voltage angle.
    """
    # Create a diagonal matrix of V/|V|
    D = np.diag(V / np.abs(V))
    
    # Compute Ybus*V and its conjugate diagonal part
    YV = Ybus.dot(V)
    D_YV = np.diag(YV.conj())
    
    # First term: diag(V/|V|) * diag(Ybus*V)*
    term1 = D.dot(D_YV)
    
    # Second term: diag(V) * (Ybus.dot(diag(V/|V|)))*
    term2 = np.diag(V).dot(Ybus.dot(np.diag(V / np.abs(V))).conj())
    
    J_dS_dVm = term1 + term2

    # For the derivative with respect to voltage angles:
    # dS/dθ = j·diag(V)·( diag(Ybus*V) - Ybus.dot(diag(V)) )*
    diff_term = np.diag(Ybus.dot(V)) - Ybus.dot(np.diag(V))
    J_dS_dTheta = 1j * np.diag(V).dot(diff_term.conj())

    return J_dS_dVm, J_dS_dTheta


def generate_Jacobian(J_dS_dVm, J_dS_dTheta, pv_index, pq_index):
    """
    Assembles the Jacobian matrix for the Newton-Raphson power flow using the derivatives.
    
    The mismatch vector F is constructed as:
        F = [ ΔP (for PV buses); ΔP (for PQ buses); ΔQ (for PQ buses) ]
    Hence, the Jacobian is partitioned as:
        J = [ J_11    J_12 ]
            [ J_21    J_22 ]
    where:
        J_11 = Real( dS/dθ ) at all PV and PQ buses:  rows = pv_index ∪ pq_index, columns = pv_index ∪ pq_index.
        J_12 = Real( dS/dV ) at PV and PQ buses for PQ columns: rows = pv_index ∪ pq_index, columns = pq_index.
        J_21 = Imag( dS/dθ ) at PQ buses: rows = pq_index, columns = pv_index ∪ pq_index.
        J_22 = Imag( dS/dV ) at PQ buses: rows = pq_index, columns = pq_index.
    
    Parameters:
        J_dS_dVm (ndarray): N×N complex matrix of derivatives of S with respect to voltage magnitude.
        J_dS_dTheta (ndarray): N×N complex matrix of derivatives of S with respect to voltage angle.
        pv_index (array_like): Indices of PV buses.
        pq_index (array_like): Indices of PQ buses.
    
    Returns:
        J (ndarray): The assembled Jacobian matrix.
    """
    # Combine PV and PQ indices for the active power part
    pvpq_ind = np.concatenate((pv_index, pq_index))
    
    # Extract sub-matrices:
    J_11 = np.real(J_dS_dTheta[np.ix_(pvpq_ind, pvpq_ind)])
    J_12 = np.real(J_dS_dVm[np.ix_(pvpq_ind, pq_index)])
    J_21 = np.imag(J_dS_dTheta[np.ix_(pq_index, pvpq_ind)])
    J_22 = np.imag(J_dS_dVm[np.ix_(pq_index, pq_index)])
    
    # Assemble the Jacobian as a block matrix
    J = np.block([[J_11, J_12],
                  [J_21, J_22]])
    
    return J


def Update_Voltages(dx, V, pv_index, pq_index):
    """
    Update the bus voltage vector V given the Newton-Raphson update dx.

    The update dx contains:
      - Δθ for all PV and PQ buses (first n_active entries, where
        n_active = len(pv_index) + len(pq_index))
      - ΔV for PQ buses (next len(pq_index) entries)
    
    The voltage at bus i is represented as:
        V[i] = Vm[i] * exp(j * θ[i])
    The update is applied as:
        - For PV and PQ buses: θ_new = θ_old + Δθ
        - For PQ buses:     Vm_new = Vm_old + ΔV
        - For other buses (e.g., reference): no update is applied.

    Parameters:
        dx (ndarray): Newton-Raphson update vector.
        V (ndarray): Current bus voltage vector (complex values).
        pv_index (list or ndarray): Indices of PV buses.
        pq_index (list or ndarray): Indices of PQ buses.

    Returns:
        V (ndarray): Updated bus voltage vector.
    """
    # Extract current voltage magnitudes and angles
    Vm = np.abs(V)
    theta = np.angle(V)
    
    # Total number of buses that will have an angle update (PV and PQ buses)
    n_active = len(pv_index) + len(pq_index)
    
    # Partition the update vector:
    # First part: Δθ for active buses
    dtheta = dx[:n_active]
    # Second part: ΔVm for PQ buses
    dVm = dx[n_active:]
    
    # Create a copy of the current voltage parameters for updating
    new_theta = theta.copy()
    new_Vm = Vm.copy()
    
    # Combine indices for buses with angle updates (PV and PQ)
    active_index = np.concatenate((pv_index, pq_index))
    
    # Update angles for PV and PQ buses
    new_theta[active_index] = theta[active_index] + dtheta
    
    # Update voltage magnitudes only for PQ buses
    new_Vm[pq_index] = Vm[pq_index] + dVm
    
    # Form the new voltage vector using updated magnitudes and angles
    V_new = new_Vm * np.exp(1j * new_theta)
    return V_new


def DisplayResults(V, lnd):
    """
    Display the results of the power flow calculation.
    
    This function prints two tables:
      1. Bus results:
         - Bus number, label, voltage magnitude (pu) and angle (deg)
         - Active and reactive power injections (generation and load)
           Generation is computed as S_gen = S_inj + S_LD, where S_inj = V * conj(Ybus @ V)
      2. Branch flows:
         - For each branch, the "from" and "to" bus numbers and the injected
           active/reactive powers at the branch ends.
    
    The input lnd is expected to be a tuple containing the following elements:
        Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels, S_LD, 
        MVA_base, V0, pq_index, pv_index, ref
    """


    N = len(V)  # number of buses
    num_branches = len(lnd.br_f)

    # Compute bus injection using the network equation: S_inj = V * conj(Ybus @ V)
    S_inj = V * np.conjugate(lnd.Ybus @ V)
    # Compute generation assuming S_inj = S_gen - S_LD  =>  S_gen = S_inj + S_LD
    S_gen = S_inj + lnd.S_LD

    # Header for Bus Results
    print("=" * 80)
    print("|{:^66}|".format("Bus results"))
    print("=" * 80)
    header = ("{:<5} {:<10} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9}"
              .format("Bus", "Label", "Mag(pu)", "Ang(deg)",
                      "Gen P(pu)", "Gen Q(pu)", "Load P(pu)", "Load Q(pu)"))
    print(header)
    print("-" * 70)
    # Loop over buses
    for i in range(N):
        bus_num = i + 1  # if bus numbering is sequential
        label = lnd.bus_labels[i]
        Vm = np.abs(V[i])
        theta = np.degrees(np.angle(V[i]))
        P_gen = S_gen[i].real
        Q_gen = S_gen[i].imag
        P_load = lnd.S_LD[i].real
        Q_load = lnd.S_LD[i].imag
        line = ("{:<5d} {:<10} {:>8.3f} {:>8.2f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}"
                .format(bus_num, label, Vm, theta, P_gen, Q_gen, P_load, Q_load))
        print(line)
    print("=" * 70)
    print()

    # Compute branch flows. For each branch, calculate the injection from the "from" and "to" ends.
    print("=" * 70)
    print("|{:^66}|".format("Branch flow"))
    print("=" * 70)
    branch_header = ("{:<5} {:<5} {:<5} {:>9} {:>9} {:>9} {:>9}"
                     .format("Br#", "From", "To", "P_from", "Q_from", "P_to", "Q_to"))
    print(branch_header)
    print("-" * 70)
    for i in range(num_branches):
        # Bus indices for this branch (in our array space)
        from_bus = lnd.br_f[i]
        to_bus = lnd.br_t[i]
        # Calculate injection at the "from" end: S_from = V[from] * conj(Y_fr[i,:] @ V)
        S_from = V[from_bus] * np.conjugate(np.dot(lnd.Y_fr[i, :], V))
        # Similarly, injection at the "to" end: S_to = V[to_bus] * np.conjugate(np.dot(Y_to[i, :], V))
        S_to = V[to_bus] * np.conjugate(np.dot(lnd.Y_to[i, :], V))
        line = ("{:<5d} {:<5d} {:<5d} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}"
                .format(i+1, from_bus+1, to_bus+1,
                        S_from.real, S_from.imag, S_to.real, S_to.imag))
        print(line)
    print("=" * 70)

def DisplayResults_and_loading(V, lnd):
    """
    Display power flow results along with loading levels for generators and branches.
    
    This function prints two tables:
      1. Bus results:
         - Bus number, label, voltage magnitude (pu) and angle (deg)
         - Active and reactive power injections (generation and load)
         - Generator loading in percentage.
           Loading is computed as:
                100 * |S_gen| / ( (Gen_rating (MVA) / MVA_base) )
           where S_gen = S_inj + S_LD.
           Only buses with a generator (provided in Gen_rating) are loaded.
           
      2. Branch flows:
         - For each branch, the "from" and "to" bus numbers, the injected active/reactive powers,
           and the branch loading percentages at the "from" and "to" ends.
           Loading is computed as:
                100 * |S_flow| / ( Br_rating (MVA) / MVA_base )
    
    The network data tuple lnd is expected to contain:
      (Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels, Sbus, S_LD,
       MVA_base, V0, pq_index, pv_index, ref, Gen_rating, Br_rating)
       
    Here:
      - Gen_rating is a list of tuples: (bus_label, MVA rating)
      - Br_rating is a list of tuples: (from_bus, to_bus, MVA rating)
      
    Note: Bus numbers for branches are assumed to be 1-based, and the bus labels
    in Gen_rating must match the entries in bus_labels.
    """

    N = len(V)          # number of buses
    num_branches = len(lnd.branch_from)

    # Create a dictionary for generator ratings keyed by bus label.
    # e.g., {'BUS1HV': rating1, 'BUS2HV': rating2, ...}
    gen_rating_dict = {bus_label: rating for (bus_label, rating, Q_max, Q_min) in lnd.gen_rating}

    # Create a dictionary for branch ratings keyed by (from_bus, to_bus) tuple (both 1-based)
    br_rating_dict = {(fb, tb,id): rating for (fb, tb,id, rating) in lnd.branch_rating}

    # Create a dictionary for transformer ratings keyed by (from_bus, to_bus) tuple (both 1-based)
    Trans_rat_dict ={(fb, tb,id): rating for (fb, tb,id, rating) in lnd.tran_rating}

    # Compute bus injection: S_inj = V * conj(Ybus @ V)
    S_inj = V * np.conjugate(lnd.Ybus @ V)
    # Compute generation at each bus: S_gen = S_inj + S_load
    S_gen = S_inj + lnd.S_load

    # Display Bus Results with Generator Loading
    print("=" * 140)
    print("|{:^136}|".format("Bus Results with Generator Loading"))
    print("=" * 140)
    bus_header = ("{:<8} {:<15} {:>12} {:>12} {:>15} {:>15} {:>20} {:>15} {:>15}"
                  .format("Bus", "Label", "Mag(pu)", "Ang(deg)",
                          "Gen P(pu)", "Gen Q(pu)", "Gen Load(%)",
                          "Load P(pu)", "Load Q(pu)"))
    print(bus_header)
    print("-" * 140)
    for i in range(N):
        bus_num = lnd.bus_numbers[i]  # bus numbering is 1-based in output
        label = lnd.bus_labels[i]
        Vm = np.abs(V[i])
        theta = np.degrees(np.angle(V[i]))
        P_gen = S_gen[i].real
        Q_gen = S_gen[i].imag
        P_load = lnd.S_load[i].real
        Q_load = lnd.S_load[i].imag

        # Look up the generator rating using the bus label.
        if bus_num in gen_rating_dict and gen_rating_dict[bus_num] > 0:
            # Convert the generator rating from MVA to per unit.
            gen_rating_pu = gen_rating_dict[bus_num] / lnd.MVA_base
            gen_loading_pct = 100 * (np.abs(S_gen[i]) / gen_rating_pu)
            #print(f"gen_rating_pu: {gen_rating_pu}")
        else:
            gen_loading_pct = 0.0
            #print(f"no gen_rating_pu!")

        line = ("{:<8d} {:<15} {:>12.3f} {:>12.2f} {:>15.3f} {:>15.3f} {:>20.3f} {:>15.3f} {:>15.3f}"
                .format(bus_num, label, Vm, theta,
                        P_gen, Q_gen, gen_loading_pct,
                        P_load, Q_load))
        print(line)
    print("=" * 140)
    print()

    # Display Branch Flow Results with Loading Percentages
    print("=" * 140)
    print("|{:^136}|".format("Branch Flow with Loading Percentages"))
    print("=" * 140)
    branch_header = ("{:<8} {:<8} {:<8} {:>15} {:>15} {:>20} {:>15} {:>15} {:>20}"
                     .format("Br#", "From", "To", "P_from", "Q_from", "Load(%) from",
                             "P_to", "Q_to", "Load(%) to"))
    print(branch_header)
    print("-" * 140)
    for i in range(num_branches):
        # Retrieve bus indices (0-based) for branch calculations.
        from_idx = lnd.branch_from[i]
        to_idx = lnd.branch_to[i]
        
        from_bus, to_bus, ID = lnd.bus_pairs[i]
        
        # Calculate branch flows at the "from" and "to" ends.
        S_from = V[from_idx] * np.conjugate(np.dot(lnd.Y_from[i, :], V))
        S_to = V[to_idx] * np.conjugate(np.dot(lnd.Y_to[i, :], V))
        
        # Look up the branch rating using the tuple (from_bus, to_bus)
        if (from_bus, to_bus, ID) in br_rating_dict and br_rating_dict[(from_bus, to_bus, ID)] > 0:
            # Convert branch rating from MVA to per unit.
            br_rating_pu = br_rating_dict[(from_bus, to_bus,ID)] / lnd.MVA_base
            load_from_pct = 100 * (np.abs(S_from) / br_rating_pu)
            load_to_pct   = 100 * (np.abs(S_to) / br_rating_pu)
        elif (from_bus,to_bus, ID)in Trans_rat_dict and Trans_rat_dict[(from_bus,to_bus, ID)] > 0:
            # Convert branch rating from MVA to per unit.
            trans_rating_pu = Trans_rat_dict[(from_bus, to_bus, ID)] / lnd.MVA_base
            load_from_pct = 100 * (np.abs(S_from) / trans_rating_pu)
            load_to_pct   = 100 * (np.abs(S_to) / trans_rating_pu)
        else:
            load_from_pct = load_to_pct = 0.0


        line = ("{:<8d} {:<8d} {:<8d} {:>15.3f} {:>15.3f} {:>20.3f} {:>15.3f} {:>15.3f} {:>20.3f}"
                .format(i+1, from_bus, to_bus,
                        S_from.real, S_from.imag, load_from_pct,
                        S_to.real, S_to.imag, load_to_pct))
        print(line)
    print("=" * 140)
