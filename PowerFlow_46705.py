"""
46705 - Power Grid Analysis
This file contains the definitions of the functions needed to
carry out Power Flow calculations in python.

How to carry out Power Flow in a new *.py file? 
See the example in table 1 in the assignment text
"""

import numpy as np
import LoadNetworkData as load

# 1. the PowerFlowNewton() function
def PowerFlowNewton(Ybus, Sbus, V0, pv_index, pq_index, max_iter, err_tol, print_progress=True):
    """
    Solve the power flow equations using the Newton-Raphson method.

    Parameters:
        Ybus (ndarray): N x N bus admittance matrix.
        Sbus (ndarray): N x 1 specified complex power injection vector (in pu).
        V0 (ndarray): Initial bus voltage vector (complex values).
        pv_index (list or ndarray): Indices of PV buses.
        pq_index (list or ndarray): Indices of PQ buses.
        max_iter (int): Maximum number of iterations.
        err_tol (float): Tolerance for convergence.
        print_progress (bool): Flag to print progress information.

    Returns:
        tuple: (V, success, n) where:
            V (ndarray): Final bus voltage vector (complex).
            success (int): 1 if the solution converged, 0 otherwise.
            n (int): Number of iterations used.
    """
    # Initialization of the status flag and iteration counter
    success = 0
    n = 0
    V = V0.copy()
    
    if print_progress:
        print(' iteration maximum P & Q mismatch (pu)')
        print(' --------- ---------------------------')
    
    # Determine mismatch between initial guess and specified power injections
    F = calculate_F(Ybus, Sbus, V, pv_index, pq_index)
    
    # Check if the desired tolerance is reached
    success = CheckTolerance(F, n, err_tol, print_progress)
    
    # Start the Newton-Raphson iteration loop
    while (not success) and (n < max_iter):
        n += 1  # Increment iteration counter
        
        # Compute derivatives and generate the Jacobian matrix
        J_dS_dVm, J_dS_dTheta = generate_Derivatives(Ybus, V)
        J = generate_Jacobian(J_dS_dVm, J_dS_dTheta, pv_index, pq_index)
        
        # Compute the update step dx
        dx = np.linalg.solve(J, F)
        
        # Update voltages and re-calculate mismatch F
        V = Update_Voltages(dx, V, pv_index, pq_index)
        F = calculate_F(Ybus, Sbus, V, pv_index, pq_index)
        
        # Check if the updated solution meets the tolerance
        success = CheckTolerance(F, n, err_tol, print_progress)
    
    # Display convergence message if required
    if success:
        if print_progress:
            print('The Newton-Raphson Power Flow Converged in %d iterations!' % n)
    else:
        if print_progress:
            print('No Convergence !!!\nStopped after %d iterations without solution...' % n)
    
    return V, success, n



# 2. the calculate_F() function
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
    I = Ybus @ V
    S_calc = V * np.conjugate(I)
    
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


# 3. the CheckTolerance() function
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
    max_mismatch = np.max(np.abs(F))
    
    # Optionally print progress information
    if print_progress:
        print(f"Iteration {n}: maximum mismatch = {max_mismatch:.6f} (tolerance = {err_tol})")
    
    # If the maximum mismatch is below the tolerance, consider the solution converged
    success = 1 if max_mismatch < err_tol else 0
    
    return success

# 4. the generate_Derivatives() function
def generate_Derivatives(Ybus, V):
    """
    Calculate the partial derivatives of the active and reactive power injections
    with respect to voltage magnitude and angle.

    The power injections are given by:
        P_i = sum_j [ Vm_i Vm_j (G_ij cos(θ_i-θ_j) + B_ij sin(θ_i-θ_j)) ]
        Q_i = sum_j [ Vm_i Vm_j (G_ij sin(θ_i-θ_j) - B_ij cos(θ_i-θ_j)) ]

    The derivatives are:
      For i ≠ j:
        dP_i/dVm_j = Vm_i (G_ij cos(θ_i-θ_j) + B_ij sin(θ_i-θ_j))
        dP_i/dθ_j = Vm_i Vm_j ( -G_ij sin(θ_i-θ_j) + B_ij cos(θ_i-θ_j) )
        dQ_i/dVm_j = Vm_i (G_ij sin(θ_i-θ_j) - B_ij cos(θ_i-θ_j))
        dQ_i/dθ_j = -Vm_i Vm_j (G_ij cos(θ_i-θ_j) + B_ij sin(θ_i-θ_j))
      For i = j:
        dP_i/dVm_i = 2 Vm_i G_ii + sum_{j≠i} Vm_j (G_ij cos(θ_i-θ_j) + B_ij sin(θ_i-θ_j))
        dP_i/dθ_i = -sum_{j≠i} Vm_i Vm_j ( -G_ij sin(θ_i-θ_j) + B_ij cos(θ_i-θ_j) )
        dQ_i/dVm_i = -2 Vm_i B_ii + sum_{j≠i} Vm_j (G_ij sin(θ_i-θ_j) - B_ij cos(θ_i-θ_j))
        dQ_i/dθ_i = -sum_{j≠i} Vm_i Vm_j (G_ij cos(θ_i-θ_j) + B_ij sin(θ_i-θ_j))

    Returns:
        J_dS_dVm: (2N x N) matrix containing [dP/dVm; dQ/dVm]
        J_dS_dTheta: (2N x N) matrix containing [dP/dθ; dQ/dθ]
    """
    N = len(V)
    Vm = np.abs(V)
    theta = np.angle(V)
    G = Ybus.real
    B = Ybus.imag

    # Initialize derivative matrices
    dP_dVm = np.zeros((N, N))
    dP_dTheta = np.zeros((N, N))
    dQ_dVm = np.zeros((N, N))
    dQ_dTheta = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                # Diagonal elements
                # dP/dVm
                sum_term_P = 0
                sum_term_Q = 0
                for k in range(N):
                    if k != i:
                        angle_diff = theta[i] - theta[k]
                        sum_term_P += Vm[k] * (G[i, k] * np.cos(angle_diff) + B[i, k] * np.sin(angle_diff))
                        sum_term_Q += Vm[k] * (G[i, k] * np.sin(angle_diff) - B[i, k] * np.cos(angle_diff))
                dP_dVm[i, i] = 2 * Vm[i] * G[i, i] + sum_term_P
                dQ_dVm[i, i] = -2 * Vm[i] * B[i, i] + sum_term_Q

                # dP/dTheta
                sum_term = 0
                for k in range(N):
                    if k != i:
                        angle_diff = theta[i] - theta[k]
                        sum_term += Vm[i] * Vm[k] * (-G[i, k] * np.sin(angle_diff) + B[i, k] * np.cos(angle_diff))
                dP_dTheta[i, i] = sum_term  # Note: this sum is negative of the off-diagonals
                # dQ/dTheta
                sum_term = 0
                for k in range(N):
                    if k != i:
                        angle_diff = theta[i] - theta[k]
                        sum_term += -Vm[i] * Vm[k] * (G[i, k] * np.cos(angle_diff) + B[i, k] * np.sin(angle_diff))
                dQ_dTheta[i, i] = sum_term

            else:
                angle_diff = theta[i] - theta[j]
                # Off-diagonal elements
                dP_dVm[i, j] = Vm[i] * (G[i, j] * np.cos(angle_diff) + B[i, j] * np.sin(angle_diff))
                dP_dTheta[i, j] = Vm[i] * Vm[j] * (-G[i, j] * np.sin(angle_diff) + B[i, j] * np.cos(angle_diff))
                dQ_dVm[i, j] = Vm[i] * (G[i, j] * np.sin(angle_diff) - B[i, j] * np.cos(angle_diff))
                dQ_dTheta[i, j] = -Vm[i] * Vm[j] * (G[i, j] * np.cos(angle_diff) + B[i, j] * np.sin(angle_diff))

    # Assemble derivative matrices into two blocks:
    # First N rows correspond to active power derivatives, next N rows to reactive power.
    J_dS_dVm = np.vstack((dP_dVm, dQ_dVm))
    J_dS_dTheta = np.vstack((dP_dTheta, dQ_dTheta))

    return J_dS_dVm, J_dS_dTheta



# 5. the generate_Jacobian() function
def generate_Jacobian(J_dS_dVm, J_dS_dTheta, pv_index, pq_index):
    """
    Assemble the Jacobian matrix for the Newton-Raphson power flow.

    The Jacobian is constructed from the partial derivatives:
        J_dS_dTheta: 2N x N matrix, where:
            - The first N rows are dP/dθ (active power derivatives with respect to angles)
            - The last N rows are dQ/dθ (reactive power derivatives with respect to angles)
        J_dS_dVm: 2N x N matrix, where:
            - The first N rows are dP/dV (active power derivatives with respect to voltage magnitudes)
            - The last N rows are dQ/dV (reactive power derivatives with respect to voltage magnitudes)

    The mismatch vector F is arranged as:
        F = [ ΔP (for all PV and PQ buses); ΔQ (for PQ buses) ]
    
    Therefore, the Jacobian is partitioned as:
        J = [ J1   J2 ]
            [ J3   J4 ]
    where:
        J1 = dP/dθ for PV and PQ buses (rows from first block, columns for active buses)
        J2 = dP/dV for PQ buses (rows from first block, columns for PQ buses)
        J3 = dQ/dθ for PQ buses (rows from second block, columns for active buses)
        J4 = dQ/dV for PQ buses (rows from second block, columns for PQ buses)
    
    Parameters:
        J_dS_dVm (ndarray): 2N x N matrix of derivatives with respect to voltage magnitude.
        J_dS_dTheta (ndarray): 2N x N matrix of derivatives with respect to voltage angle.
        pv_index (list or ndarray): Indices of PV buses.
        pq_index (list or ndarray): Indices of PQ buses.
    
    Returns:
        J (ndarray): The assembled Jacobian matrix.
    """
    # Combine PV and PQ indices for the active power mismatch part.
    active_index = np.concatenate((pv_index, pq_index))
    
    # Total number of buses
    N = J_dS_dVm.shape[1]
    
    # Extract submatrices from the derivatives.
    # For active power mismatches:
    # J1: dP/dθ for active buses (from first N rows of J_dS_dTheta)
    J1_full = J_dS_dTheta[:N, :]  # shape: N x N
    J1 = J1_full[np.ix_(active_index, active_index)]
    
    # J2: dP/dV for PQ buses (from first N rows of J_dS_dVm)
    J2_full = J_dS_dVm[:N, :]  # shape: N x N
    J2 = J2_full[np.ix_(active_index, pq_index)]
    
    # For reactive power mismatches:
    # J3: dQ/dθ for PQ buses (from last N rows of J_dS_dTheta)
    J3_full = J_dS_dTheta[N:, :]  # shape: N x N
    J3 = J3_full[np.ix_(pq_index, active_index)]
    
    # J4: dQ/dV for PQ buses (from last N rows of J_dS_dVm)
    J4_full = J_dS_dVm[N:, :]  # shape: N x N
    J4 = J4_full[np.ix_(pq_index, pq_index)]
    
    # Assemble the full Jacobian using block concatenation.
    top = np.hstack((J1, J2))
    bottom = np.hstack((J3, J4))
    J = np.vstack((top, bottom))
    
    return J



# 6. the Update_Voltages() function
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




####################################################
#  Displaying the results in the terminal window   #
####################################################
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
    import numpy as np

    # Unpack network data from lnd
    (Ybus, Y_fr, Y_to, br_f, br_t, buscode, bus_labels, S_LD,
     MVA_base, V0, pq_index, pv_index, ref) = lnd

    N = len(V)  # number of buses
    num_branches = len(br_f)

    # Compute bus injection using the network equation: S_inj = V * conj(Ybus @ V)
    S_inj = V * np.conjugate(Ybus @ V)
    # Compute generation assuming S_inj = S_gen - S_LD  =>  S_gen = S_inj + S_LD
    S_gen = S_inj + S_LD

    # Header for Bus Results
    print("=" * 70)
    print("|{:^66}|".format("Bus results"))
    print("=" * 70)
    header = ("{:<5} {:<10} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9}"
              .format("Bus", "Label", "Mag(pu)", "Ang(deg)",
                      "Gen P(pu)", "Gen Q(pu)", "Load P(pu)", "Load Q(pu)"))
    print(header)
    print("-" * 70)
    # Loop over buses
    for i in range(N):
        bus_num = i + 1  # if bus numbering is sequential
        label = bus_labels[i]
        Vm = np.abs(V[i])
        theta = np.degrees(np.angle(V[i]))
        P_gen = S_gen[i].real
        Q_gen = S_gen[i].imag
        P_load = S_LD[i].real
        Q_load = S_LD[i].imag
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
        from_bus = br_f[i]
        to_bus = br_t[i]
        # Calculate injection at the "from" end: S_from = V[from] * conj(Y_fr[i,:] @ V)
        S_from = V[from_bus] * np.conjugate(np.dot(Y_fr[i, :], V))
        # Similarly, injection at the "to" end: S_to = V[to_bus] * np.conjugate(np.dot(Y_to[i, :], V))
        S_to = V[to_bus] * np.conjugate(np.dot(Y_to[i, :], V))
        line = ("{:<5d} {:<5d} {:<5d} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}"
                .format(i+1, from_bus+1, to_bus+1,
                        S_from.real, S_from.imag, S_to.real, S_to.imag))
        print(line)
    print("=" * 70)
