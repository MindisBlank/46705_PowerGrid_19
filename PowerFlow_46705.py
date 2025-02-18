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

# 1. the PowerFlowNewton() function
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
    max_mismatch = np.linalg.norm(F,np.inf)
    # Optionally print progress information
    if print_progress:
        print(f"Iteration {n}: maximum mismatch = {max_mismatch:.6f} (tolerance = {err_tol})")
    
    # If the maximum mismatch is below the tolerance, consider the solution converged
    success = 1 if max_mismatch < err_tol else 0
    
    return success

# 4. the generate_Derivatives() function
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
def DisplayResults(V,lnd):

    return
