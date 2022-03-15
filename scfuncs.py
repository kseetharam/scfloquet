import numpy as np
import xarray as xr
from scipy.integrate import simps, romb
from scipy.special import binom
from scipy.stats import uniform, randint, multivariate_normal
import mpmath as mpm
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from timeit import default_timer as timer
from scipy.optimize import minimize, root_scalar
from copy import copy, deepcopy
import qutip as qt


# ---- HELPER FUNCTIONS ----


def get_U(diagH, P, t):
    # Return the evolution operator at time t
    diagU = np.exp(-1j * t * diagH)
    # return np.matmul(P, np.matmul(np.diag(diagU), np.transpose(np.conj(P))))
    return P @ np.diag(diagU) @ np.conj(P).T


def basisVector(bint, spinBasis):
    # create basis state in vector representation (=bstate) based on binary representation (=bstr) of integer representation of the state (=bint)
    bstr = format(bint, '0' + str(int(spinBasis.L)) + 'b'); i0 = spinBasis.index(bstr)
    bstate = np.zeros(spinBasis.Ns, dtype=np.float64); bstate[i0] = 1
    # # Alternate way that just directly fills the appropriate vector
    # bstate = np.zeros(basis.Ns, dtype=np.float64); bstate[basis.Ns - bint - 1] = 1
    return bstate


def SzTot(spinBasis):
    # returns list containing values of the total z-magnetization for each basis vector in the spin basis. Note that this operator is diagonal in the basis.
    return np.array([bin(bint).count("1") - (spinBasis.L / 2) for bint in spinBasis.states])  # total z-magnetization is (N_up - N_down)/2 = (N_up - (L - N_up))/2.


def aveSzTot(psi_t, Sz_Tot, ShotNoise, N_ShotNoise):
    # compute expectation value of total z-magnetization in the time-evolved state. psi_t is either a matrix (with dim > 1) where each column is a state at a different time or psi_t is a single state (with dim = 1) at a fixed time
    # in the case where psi_t is a matrix: yields a vector of expectation values for each time sample as |psi_t|^2 givest coefficients of each state for each time step which is a matrix; we multiply each column by the magnetization of each basis state as each column corresponds to the basis expansion of a state at a different time)
    # in the case where psi_t is a vector: yeilds a number which is the expectation value of the total z-magnetization in that state
    # ShotNoise is a boolean describing whether we generate a sample mean for the expectation value or the true expectation value. In the case of generating the sample mean, N_ShotNoise tells us how many samples to take.
    state_prob = np.abs(psi_t)**2
    if psi_t.ndim > 1:
        if ShotNoise:
            return np.apply_along_axis(sampleMean, 0, state_prob, Sz_Tot, N_ShotNoise, np.random.default_rng())
        else:
            return np.sum(np.multiply(Sz_Tot[:, None], state_prob), axis=0)
    else:
        if ShotNoise:
            return sampleMean(state_prob, Sz_Tot, N_ShotNoise, np.random.default_rng())
        else:
            return np.dot(Sz_Tot, state_prob)


# def aveSzTot(psi_t, Sz_Tot, ShotNoise, N_ShotNoise):
#     # compute expectation value of total z-magnetization in the time-evolved state. psi_t is either a matrix (with dim > 1) where each column is a state at a different time or psi_t is a single state (with dim = 1) at a fixed time
#     # in the case where psi_t is a matrix: yields a vector of expectation values for each time sample as |psi_t|^2 givest coefficients of each state for each time step which is a matrix; we multiply each column by the magnetization of each basis state as each column corresponds to the basis expansion of a state at a different time)
#     # in the case where psi_t is a vector: yeilds a number which is the expectation value of the total z-magnetization in that state
#     if psi_t.ndim > 1:
#         return np.sum(np.multiply(Sz_Tot[:, None], np.abs(psi_t)**2), axis=0)
#     else:
#         return np.dot(Sz_Tot, np.abs(psi_t)**2)


def sampleMean(probDist, Sz_Tot, NSamples, rng):
    # takes an array of discrete probabilities (probDist) which represent probabilities for each of the magnetizations in the array Sz_Tot. Also takes a random number generator
    # Then draws NSamples of magnetization from this probability distribution and returns the sample mean
    probDist_temp = probDist + (1 - np.sum(probDist)); pD_ind = np.argwhere(probDist_temp > 0)[0][0]
    probDist[pD_ind] = probDist[pD_ind] + (1 - np.sum(probDist))  # fixes probability distribution if it's normalization is slightly off (rng.choice seems to have a problem with normalization of even ~0.99995)
    return np.mean(rng.choice(a=Sz_Tot, size=NSamples, p=probDist))


def spectFunc(St, tVals, decayRate):
    # takes in a function S(t) and time values (in s) and outputs A(\omega) and frequency values (in Hz)
    # decayRate = 4 * (1 / np.max(tVals))
    # decayRate = 1
    # print('Decay constant: {0}'.format(decay_const))
    St_fixed = deepcopy(St); St_fixed[0] = 0.5 * St_fixed[0]  # fixes double counting in the FFT so that the spectrum correctly starts at zero instead of having a constant shift
    decay_window = np.exp(-1 * decayRate * tVals)
    nsamples = St_fixed.size; dt = tVals[1] - tVals[0]
    fVals = np.fft.fftshift(np.fft.fftfreq(nsamples, dt))
    FTVals = np.fft.fftshift(np.fft.fft(St_fixed * decay_window) / nsamples)
    Aw = np.real(FTVals)
    # Aw = FTVals
    return Aw, fVals


def apodize_g1d(St, k):
    St_fixed = deepcopy(St); St_fixed[0] = 0.5 * St_fixed[0]  # fixes double counting in the FFT so that the spectrum correctly starts at zero instead of having a constant shift
    x = np.linspace(0, 1, St_fixed.size)
    return St_fixed * np.exp(-1 * k * x**2)


def apodize_exp1d(St, k):
    St_fixed = deepcopy(St); St_fixed[0] = 0.5 * St_fixed[0]  # fixes double counting in the FFT so that the spectrum correctly starts at zero instead of having a constant shift
    x = np.linspace(0, 1, St_fixed.size)
    return St_fixed * np.exp(-1 * k * x)


def spectFunc_pa(St, tVals):
    fVals = np.fft.fftshift(np.fft.fftfreq(St.size, tVals[1] - tVals[0]))
    FTVals = np.fft.fftshift(np.fft.fft(St) / St.size)
    return np.real(FTVals), fVals


def hamiltonianListToMatrix(HParams, NSpins, fix_factor_of_2):
    # the fix_factor_of_2 corrects for the factor of 2 added to Jij in HParams to output the true Hamiltonian matrix
    if fix_factor_of_2:
        f2 = 2
    else:
        f2 = 1

    hamMat = np.zeros((NSpins, NSpins))
    [JijList, hiList] = HParams
    for indh, hTerm in enumerate(JijList):
        [Jij, i, j] = hTerm
        hamMat[i, j] = Jij / f2; hamMat[j, i] = Jij / f2
    for indh, hTerm in enumerate(hiList):
        [hi, i] = hTerm
        hamMat[i, i] = hi
    return hamMat


# ---- ED FUNCTIONS ----


def fullHamiltonian(spinBasis, JijList, hiList, chemicalShifts=True):
    # takes parameters Jij and hi (in Hz) and outputs the Hamiltonian
    # chemicalShift is a flag that decides whether we include the hi terms or just the interactions Jij
    static = [["xx", JijList], ["yy", JijList], ["zz", JijList]]; dynamic = []
    # static = [["xx", JijList]]; dynamic = []; print('zz')
    if chemicalShifts:
        static.append(["x", hiList])
    return hamiltonian(static, dynamic, basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False)


def scHamParams(JijList, hiList):
    JijList_sc = []
    hiList_sc = []
    for tup in JijList:
        [val, i, j] = tup
        JijList_sc.append([3 * val / 2, i, j])
    for tup in hiList:
        [val, i] = tup
        hiList_sc.append([3 * val, i])
    return JijList_sc, hiList_sc


def scHamiltonian(spinBasis, JijList, hiList, chemicalShifts=True):
    # takes parameters Jij and hi (in Hz) and outputs the Hamiltonian
    # chemicalShift is a flag that decides whether we include the hi terms or just the interactions Jij
    JijList_sc, hiList_sc = scHamParams(JijList, hiList)
    static = [["xx", JijList_sc], ["yy", JijList_sc]]; dynamic = []
    # static = [["xx", JijList]]; dynamic = []; print('zz')
    if chemicalShifts:
        static.append(["z", hiList_sc])
    return hamiltonian(static, dynamic, basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False)

def getSpinOperators(N):
    si = qt.qeye(2); sx = 0.5 * qt.sigmax(); sy = 0.5 * qt.sigmay(); sz = 0.5 * qt.sigmaz()
    si_list = []
    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        si_list.append(qt.tensor(op_list))

        op_list[n] = sx
        sx_list.append(qt.tensor(op_list))

        op_list[n] = sy
        sy_list.append(qt.tensor(op_list))

        op_list[n] = sz
        sz_list.append(qt.tensor(op_list))
    return si_list, sx_list, sy_list, sz_list

def makeRotations(operator_list, pulseAngle):
    si_list, sx_list, sy_list, sz_list = operator_list
    N = len(si_list)

    sx_tot = 0; sy_tot = 0; sz_tot = 0
    for n in range(N):
        sx_tot += sx_list[n]
        sy_tot += sy_list[n]
        sz_tot += sz_list[n]

    Rx = (-1j*pulseAngle*sx_tot).expm()
    Ry = (-1j*pulseAngle*sy_tot).expm()
    Rz = (-1j*pulseAngle*sz_tot).expm()    

    return Rx, Ry, Rz

def makeGenerator_Hnmr(operator_list, JijList, hiList):
    si_list, sx_list, sy_list, sz_list = operator_list

    H = 0

    for tup in hiList:
        hi, i = tup
        if hi != 0:
            H += hi * sx_list[i]

    for tup in JijList:
        Jij, i, j = tup
        H += Jij * (sx_list[i] * sx_list[j] + sy_list[i] * sy_list[j] + sz_list[i] * sz_list[j])

    return H

def makeGenerator_Hsc(operator_list, JijList, hiList):
    si_list, sx_list, sy_list, sz_list = operator_list

    JijList_sc, hiList_sc = scHamParams(JijList, hiList)

    H = 0

    for tup in hiList_sc:
        hi, i = tup
        if hi != 0:
            H += hi * sz_list[i]

    for tup in JijList_sc:
        Jij, i, j = tup
        H += Jij * (sx_list[i] * sx_list[j] + sy_list[i] * sy_list[j])

    return H


def makeGenerator_Hnative(operator_list, J_S, J_I, hiList):
    # assumes nearest-neighbor interactions of qubits according to native SC Hamiltonian H = \sum_{i<j} { J_S * (S_{i}^{x}*S_{j}^{x}+S_{i}^{y}*S_{j}^{y}) + J_I * S_{i}^{z}*S_{j}^{z} } + \sum_{i} h_{I}*S_{i}^{z}

    si_list, sx_list, sy_list, sz_list = operator_list
    N = len(si_list)
    H = 0

    for hi in hiList:
        if hi != 0:
            H += hi * sz_list[i]

    # int_list = [] # list of spins which interact

    if N < 4:
        for i in np.arange(N):
            if (i + 1) < N:
                H += J_S * (sx_list[i] * sx_list[i+1] + sy_list[i] * sy_list[i+1]) + J_I * sz_list[i] * sz_list[i+1]
                # int_list.append((i,i+1))
    else:
        Nlin = int(np.sqrt(N))
        grid = np.reshape(range(N),(Nlin,Nlin))
        # print(grid)
        for i in np.arange(Nlin):
            for j in np.arange(Nlin):
                if (i + 1) < Nlin:
                    H += J_S * (sx_list[grid[i,j]] * sx_list[grid[i+1,j]] + sy_list[grid[i,j]] * sy_list[grid[i+1,j]]) + J_I * sz_list[grid[i,j]] * sz_list[grid[i+1,j]]
                    # int_list.append((grid[i,j],grid[i+1,j]))                        
                if (j + 1) < Nlin:
                    H += J_S * (sx_list[grid[i,j]] * sx_list[grid[i,j+1]] + sy_list[grid[i,j]] * sy_list[grid[i,j+1]]) + J_I * sz_list[grid[i,j]] * sz_list[grid[i,j+1]]
                    # int_list.append((grid[i,j],grid[i,j+1]))
    # print(int_list)                        

    return H

def makeGenerator_Hxxz(operator_list, J, Delta, hiList):
    # assumes nearest-neighbor interactions of qubits according to native SC Hamiltonian H = \sum_{i<j} { J_S * (S_{i}^{x}*S_{j}^{x}+S_{i}^{y}*S_{j}^{y}) + J_I * S_{i}^{z}*S_{j}^{z} } + \sum_{i} h_{I}*S_{i}^{z}

    si_list, sx_list, sy_list, sz_list = operator_list
    N = len(si_list)
    H = 0

    for hi in hiList:
        if hi != 0:
            H += hi * sz_list[i]

    # int_list = [] # list of spins which interact

    if N < 4:
        for i in np.arange(N):
            if (i + 1) < N:
                H += J * (sx_list[i] * sx_list[i+1] + sy_list[i] * sy_list[i+1]) + J * Delta * sz_list[i] * sz_list[i+1]
                # int_list.append((i,i+1))
    else:
        Nlin = int(np.sqrt(N))
        grid = np.reshape(range(N),(Nlin,Nlin))
        # print(grid)
        for i in np.arange(Nlin):
            for j in np.arange(Nlin):
                if (i + 1) < Nlin:
                    H += J * (sx_list[grid[i,j]] * sx_list[grid[i+1,j]] + sy_list[grid[i,j]] * sy_list[grid[i+1,j]]) + J*Delta * sz_list[grid[i,j]] * sz_list[grid[i+1,j]]
                    # int_list.append((grid[i,j],grid[i+1,j]))                        
                if (j + 1) < Nlin:
                    H += J * (sx_list[grid[i,j]] * sx_list[grid[i,j+1]] + sy_list[grid[i,j]] * sy_list[grid[i,j+1]]) + J*Delta * sz_list[grid[i,j]] * sz_list[grid[i,j+1]]
                    # int_list.append((grid[i,j],grid[i,j+1]))
    # print(int_list)                        

    return H


def makeGenerator_depolarize(operator_list, kappa, gamma):
    if gamma == 0:
        return []
    si_list, sx_list, sy_list, sz_list = operator_list
    N = len(si_list)
    c_op_list = []
    for i in range(N):
        c_op_list.append(kappa * np.sqrt(1 - gamma) * si_list[i])
        c_op_list.append(kappa * np.sqrt(gamma / 3) * 2 * sx_list[i])
        c_op_list.append(kappa * np.sqrt(gamma / 3) * 2 * sy_list[i])
        c_op_list.append(kappa * np.sqrt(gamma / 3) * 2 * sz_list[i])

    return c_op_list


def makeGenerator_ampdamp(operator_list, kappa, gamma):
    if gamma == 0:
        return []
    si_list, sx_list, sy_list, sz_list = operator_list
    N = len(si_list)
    c_op_list = []
    for i in range(N):
        m0 = 0.5 * (1 + np.sqrt(1 - gamma)) * si_list[i] + (1 - np.sqrt(1 - gamma)) * sz_list[i]
        m1 = np.sqrt(gamma) * sx_list[i] + 1j * np.sqrt(gamma) * sy_list[i] 
        c_op_list.append(kappa * m0)
        c_op_list.append(kappa * m1)

    return c_op_list

def makeGenerator_phasedamp(operator_list, kappa, gamma):
    if gamma == 0:
        return []
    si_list, sx_list, sy_list, sz_list = operator_list
    N = len(si_list)
    c_op_list = []
    for i in range(N):
        # m0 = 0.5 * (1 + np.sqrt(1 - gamma)) * si_list[i] + (1 - np.sqrt(1 - gamma)) * sz_list[i]
        # m1 = 0.5 * np.sqrt(gamma) * si_list[i] - np.sqrt(gamma) * sz_list[i] 
        m0 = np.sqrt(1 - gamma) * si_list[i]
        m1 = np.sqrt(gamma) * 2 * sz_list[i] 
        c_op_list.append(kappa * m0)
        c_op_list.append(kappa * m1)

    return c_op_list


def makeGenerator_nmr(operator_list, JijList, hiList, kappa, gamma_amp, gamma_phase):
    si_list, sx_list, sy_list, sz_list = operator_list

    H = makeGenerator_Hnmr(operator_list, JijList, hiList)

    c_op_list_amp = makeGenerator_ampdamp(operator_list, kappa, gamma_amp)
    c_op_list_phase = makeGenerator_phasedamp(operator_list, kappa, gamma_phase)
    c_op_list = c_op_list_amp + c_op_list_phase

    return H, c_op_list


def makeGenerator_sc(operator_list, JijList, hiList, kappa, gamma_amp, gamma_phase):
    si_list, sx_list, sy_list, sz_list = operator_list
    JijList_sc, hiList_sc = scHamParams(JijList, hiList)

    H = makeGenerator_Hsc(operator_list, JijList, hiList)

    c_op_list_amp = makeGenerator_ampdamp(operator_list, kappa, gamma_amp)
    c_op_list_phase = makeGenerator_phasedamp(operator_list, kappa, gamma_phase)
    c_op_list = c_op_list_amp + c_op_list_phase

    return H, c_op_list


def makeGenerator_native(operator_list, J_S, J_I, hiList, kappa, gamma_amp, gamma_phase):
    si_list, sx_list, sy_list, sz_list = operator_list

    H = makeGenerator_Hnative(operator_list, J_S, J_I, hiList)

    c_op_list_amp = makeGenerator_ampdamp(operator_list, kappa, gamma_amp)
    c_op_list_phase = makeGenerator_phasedamp(operator_list, kappa, gamma_phase)
    c_op_list = c_op_list_amp + c_op_list_phase

    return H, c_op_list


def makeGenerator_xxz(operator_list, J, Delta, hiList, kappa, gamma_amp, gamma_phase):
    si_list, sx_list, sy_list, sz_list = operator_list

    H = makeGenerator_Hxxz(operator_list, J, Delta, hiList)

    c_op_list_amp = makeGenerator_ampdamp(operator_list, kappa, gamma_amp)
    c_op_list_phase = makeGenerator_phasedamp(operator_list, kappa, gamma_phase)
    c_op_list = c_op_list_amp + c_op_list_phase

    return H, c_op_list



def T1sim(tgrid, kappa, gamma_amp, gamma_phase):
    si = qt.qeye(2); sx = 0.5 * qt.sigmax(); sy = 0.5 * qt.sigmay(); sz = 0.5 * qt.sigmaz()
    pulseAngle = np.pi / 2
    Rx = (-1j*pulseAngle*sx).expm(); Rx_m = (1j*pulseAngle*sx).expm()
    Ry = (-1j*pulseAngle*sy).expm(); Ry_m = (1j*pulseAngle*sy).expm()
    Rz = (-1j*pulseAngle*sz).expm(); Rz_m = (1j*pulseAngle*sz).expm()      

    operator_list = getSpinOperators(1)

    H = 0*si
    c_op_list_amp = makeGenerator_ampdamp(operator_list, kappa, gamma_amp)
    c_op_list_phase = makeGenerator_phasedamp(operator_list, kappa, gamma_phase)
    c_op_list = c_op_list_amp + c_op_list_phase
    
    spin = qt.basis(2, 0)
    rho0 = qt.ket2dm(Rx*Rx*spin)
    result = qt.mesolve(H, rho0, tgrid, c_op_list)

    state_prob = np.zeros((tgrid.size), dtype=float)
    for indt, t in enumerate(tgrid):
        rho_t = result.states[indt]
        state_prob[indt] = np.abs(np.diag(rho_t.full()))[1]

    state_prob_da = xr.DataArray(state_prob, coords=[tgrid], dims=['t'])
    data_dict = {'state_prob': state_prob_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'kappa': kappa, 'gamma_amp': gamma_amp, 'gamma_phase':gamma_phase}
    ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)

    return ds


def T2sim(tgrid, kappa, gamma_amp, gamma_phase):
    si = qt.qeye(2); sx = 0.5 * qt.sigmax(); sy = 0.5 * qt.sigmay(); sz = 0.5 * qt.sigmaz()
    pulseAngle = np.pi / 2
    Rx = (-1j*pulseAngle*sx).expm(); Rx_m = (1j*pulseAngle*sx).expm()
    Ry = (-1j*pulseAngle*sy).expm(); Ry_m = (1j*pulseAngle*sy).expm()
    Rz = (-1j*pulseAngle*sz).expm(); Rz_m = (1j*pulseAngle*sz).expm()      

    operator_list = getSpinOperators(1)

    H = 0*si
    c_op_list_amp = makeGenerator_ampdamp(operator_list, kappa, gamma_amp)
    c_op_list_phase = makeGenerator_phasedamp(operator_list, kappa, gamma_phase)
    c_op_list = c_op_list_amp + c_op_list_phase
    
    spin = qt.basis(2, 0)
    rho0 = qt.ket2dm(Rx*spin)
    result = qt.mesolve(H, rho0, tgrid, c_op_list)

    state_prob = np.zeros((tgrid.size), dtype=float)
    for indt, t in enumerate(tgrid):
        rho_t = Rx * result.states[indt] * Rx_m
        state_prob[indt] = np.abs(np.diag(rho_t.full()))[1]

    state_prob_da = xr.DataArray(state_prob, coords=[tgrid], dims=['t'])
    data_dict = {'state_prob': state_prob_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'kappa': kappa, 'gamma_amp': gamma_amp, 'gamma_phase':gamma_phase}
    ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)

    return ds


def trueSim(tgrid, spinBasis, HParams, shotNoiseParams, decayRate, posMag=True, saveAllStates=False):
    # Compute response function S(t|\theta) and spectral function using non-Trotterized dynamics
    # posMag=True does the computation for only positive magnetization basis states: this is sufficient to calculate the response function and spectrum, but not to calculate the average fidelity.
    # saveAllStates=True saves the full time-evolved system state for all time samples of S(t). If it is set to False, then it only saves the state at the last time
    ShotNoise = shotNoiseParams['ShotNoise']; N_ShotNoise = shotNoiseParams['N_ShotNoise']
    [JijList, hiList] = HParams
    H_theta = fullHamiltonian(spinBasis, JijList, hiList, chemicalShifts=True)
    N = spinBasis.L
    Sz_Tot = SzTot(spinBasis)
    if posMag:
        magMask = np.logical_not(np.isclose(Sz_Tot, 0.0, atol=1e-3)) * (Sz_Tot > 0.0)
        Sz_Tot_comp = Sz_Tot[magMask]
        basisStates_comp = spinBasis.states[magMask]
    else:
        Sz_Tot_comp = Sz_Tot
        basisStates_comp = spinBasis.states

    St_theta_Mat = np.zeros((basisStates_comp.size, tgrid.size))

    if saveAllStates:
        psiFinal_Real_da = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns, tgrid.size), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states, tgrid], dims=['basisStates_comp', 'basisStates_all', 't'])
        psiFinal_Imag_da = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns, tgrid.size), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states, tgrid], dims=['basisStates_comp', 'basisStates_all', 't'])
    else:
        psiFinal_Real_da = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states], dims=['basisStates_comp', 'basisStates_all'])
        psiFinal_Imag_da = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states], dims=['basisStates_comp', 'basisStates_all'])

    # diagH, P = H_theta.eigh()
    # print(diagH)
    for indb, bint in enumerate(basisStates_comp):
        bstate = basisVector(bint, spinBasis)
        # print(bint, Sz_tot[indb], bstate)  # integer representation of basis state, total magnetization associated with the state, and vector representation of the state

        # time-evolve basis state for all desired time samples ***If storing this array (basis.Ns x tgrid.size = N*2^N where N is number of spins) in memory is too difficult, we can compute time sample by sample (tradeoff is runtime)
        psi_t = np.array(H_theta.evolve(bstate, 0.0, tgrid))

        # psi_t_explicit = np.zeros(psi_t.shape, dtype='complex')
        # for indt, t in enumerate(tgrid):
        #     U_theta = get_U(diagH, P, t)
        #     psi_t_explicit[:, indt] = np.dot(U_theta, bstate)
        # # print(np.allclose(psi_t, psi_t_explicit))
        # # print(psi_t - psi_t_explicit)
        # psi_t = psi_t_explicit

        ave_Sz_Tot = aveSzTot(psi_t, Sz_Tot, ShotNoise, N_ShotNoise)
        St_theta_Mat[indb, :] = ave_Sz_Tot * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

        if saveAllStates:
            psiFinal_Real_da[indb] = np.real(psi_t)
            psiFinal_Imag_da[indb] = np.imag(psi_t)
        else:
            psi_t_final = psi_t[:, -1]
            psiFinal_Real_da[indb] = np.real(psi_t_final)
            psiFinal_Imag_da[indb] = np.imag(psi_t_final)

    if posMag:
        St_theta = 2 * np.sum(St_theta_Mat, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector. For some reason St_theta[0] is a little off from the sum rule (N/4) when we compute only over the positive magnetization sector...
    else:
        St_theta = np.sum(St_theta_Mat, axis=0)

    Aw_theta, fVals = spectFunc(St_theta, tgrid, decayRate)
    df = fVals[1] - fVals[0]
    # print(N / 4, St_theta[0], 0.5 * simps(Aw_theta, dx=df))  # checks sum rule (spectra Aw_theta may not satisfy it due to resolution in frequency and decay we input by hand)

    Spectrum_da = xr.DataArray(Aw_theta, coords=[fVals], dims=['f'])
    ResponseFunc_da = xr.DataArray(St_theta, coords=[tgrid], dims=['t'])
    hamiltonian_da = xr.DataArray(hamiltonianListToMatrix(HParams, spinBasis.L, True), coords=[np.arange(spinBasis.L), np.arange(spinBasis.L)], dims=['i', 'j'])

    data_dict = {'Spectrum': Spectrum_da, 'ResponseFunc': ResponseFunc_da, 'psiFinal_Real': psiFinal_Real_da, 'psiFinal_Imag': psiFinal_Imag_da, 'HamiltonianMatrix': hamiltonian_da}
    coords_dict = {'f': fVals, 't': tgrid, 'basisStates_comp': basisStates_comp, 'basisStates_all': spinBasis.states, 'i': np.arange(spinBasis.L), 'j': np.arange(spinBasis.L)}
    attrs_dict = {'Nspins': N}

    trueED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trueED_ds


def trueSim_complex(tgrid, spinBasis, HParams, shotNoiseParams, decayRate, posMag=True):
    ShotNoise = shotNoiseParams['ShotNoise']; N_ShotNoise = shotNoiseParams['N_ShotNoise']
    [JijList, hiList] = HParams
    H_theta = fullHamiltonian(spinBasis, JijList, hiList, chemicalShifts=True)
    diagH, P = H_theta.eigh()

    N = spinBasis.L

    pulseAngle = np.pi / 2

    # ham_Rx = hamiltonian([["x", [[1, i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rx, P_Rx = ham_Rx.eigh()
    # U_Rx = P_Rx @ np.diag(np.exp(-1j * pulseAngle * diagH_Rx)) @ np.conj(P_Rx).T; U_Rx_m = P_Rx @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Rx)) @ np.conj(P_Rx).T

    ham_Rx = hamiltonian([["x", [[1, i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rx, P_Rx = ham_Rx.eigh()
    ham_Ry = hamiltonian([["y", [[1, i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.complex128, check_symm=False, check_herm=False); diagH_Ry, P_Ry = ham_Ry.eigh()
    ham_Rz = hamiltonian([["z", [[1, i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rz, P_Rz = ham_Rz.eigh()

    U_Rx = P_Rx @ np.diag(np.exp(-1j * pulseAngle * diagH_Rx)) @ np.conj(P_Rx).T; U_Rx_m = P_Rx @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Rx)) @ np.conj(P_Rx).T
    U_Ry = P_Ry @ np.diag(np.exp(-1j * pulseAngle * diagH_Ry)) @ np.conj(P_Ry).T; U_Ry_m = P_Ry @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Ry)) @ np.conj(P_Ry).T
    U_Rz = P_Rz @ np.diag(np.exp(-1j * pulseAngle * diagH_Rz)) @ np.conj(P_Rz).T; U_Rz_m = P_Rz @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Rz)) @ np.conj(P_Rz).T

    Sz_Tot = SzTot(spinBasis)
    if posMag:
        magMask = np.logical_not(np.isclose(Sz_Tot, 0.0, atol=1e-3)) * (Sz_Tot > 0.0)
        Sz_Tot_comp = Sz_Tot[magMask]
        basisStates_comp = spinBasis.states[magMask]
    else:
        Sz_Tot_comp = Sz_Tot
        basisStates_comp = spinBasis.states

    St_theta_Mat = np.zeros((basisStates_comp.size, tgrid.size), dtype=complex)

    for indb, bint in enumerate(basisStates_comp):
        bstate = basisVector(bint, spinBasis)

        psi_t_re = np.zeros((bstate.size, tgrid.size), dtype=complex)
        psi_t_im = np.zeros((bstate.size, tgrid.size), dtype=complex)
        for indt, t in enumerate(tgrid):
            U_theta = get_U(diagH, P, t)
            psi_t_re[:, indt] = np.dot(U_theta, bstate)
            psi_t_im[:, indt] = np.dot(U_Rx @ U_theta, bstate)

        ave_Sz_Tot_re = aveSzTot(psi_t_re, Sz_Tot, ShotNoise, N_ShotNoise)
        ave_Sz_Tot_im = aveSzTot(psi_t_im, Sz_Tot, ShotNoise, N_ShotNoise)
        ave_Sz_Tot = ave_Sz_Tot_re + 1j * ave_Sz_Tot_im
        St_theta_Mat[indb, :] = ave_Sz_Tot * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

        # St_theta_Mat_re = ave_Sz_Tot_re * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)
        # St_theta_Mat_im = ave_Sz_Tot_im * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)
        # St_theta_Mat[indb, :] = St_theta_Mat_re + 1j*St_theta_Mat_im

    if posMag:
        St_theta = 2 * np.sum(St_theta_Mat, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector. For some reason St_theta[0] is a little off from the sum rule (N/4) when we compute only over the positive magnetization sector...
    else:
        St_theta = np.sum(St_theta_Mat, axis=0)

    Aw_theta, fVals = spectFunc(St_theta, tgrid, decayRate)
    df = fVals[1] - fVals[0]
    # print(N / 4, St_theta[0], 0.5 * simps(Aw_theta, dx=df))  # checks sum rule (spectra Aw_theta may not satisfy it due to resolution in frequency and decay we input by hand)

    Spectrum_da = xr.DataArray(Aw_theta, coords=[fVals], dims=['f'])
    ResponseFunc_Real_da = xr.DataArray(np.real(St_theta), coords=[tgrid], dims=['t'])
    ResponseFunc_Imag_da = xr.DataArray(np.imag(St_theta), coords=[tgrid], dims=['t'])

    hamiltonian_da = xr.DataArray(hamiltonianListToMatrix(HParams, spinBasis.L, True), coords=[np.arange(spinBasis.L), np.arange(spinBasis.L)], dims=['i', 'j'])

    data_dict = {'Spectrum': Spectrum_da, 'ResponseFunc_Real': ResponseFunc_Real_da, 'ResponseFunc_Imag': ResponseFunc_Imag_da, 'HamiltonianMatrix': hamiltonian_da}
    coords_dict = {'f': fVals, 't': tgrid, 'basisStates_comp': basisStates_comp, 'basisStates_all': spinBasis.states, 'i': np.arange(spinBasis.L), 'j': np.arange(spinBasis.L)}
    attrs_dict = {'Nspins': N}

    trueED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trueED_ds

def trueSim_complex_aveHam(tgrid, spinBasis, HParams, shotNoiseParams, decayRate, posMag=True):
    ShotNoise = shotNoiseParams['ShotNoise']; N_ShotNoise = shotNoiseParams['N_ShotNoise']
    [JijList, hiList] = HParams
    H_theta = scHamiltonian(spinBasis, JijList, hiList, chemicalShifts=True)
    diagH, P = H_theta.eigh()

    N = spinBasis.L

    pulseAngle = np.pi / 2

    ham_Rx = hamiltonian([["x", [[1, i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rx, P_Rx = ham_Rx.eigh()
    ham_Ry = hamiltonian([["y", [[1, i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.complex128, check_symm=False, check_herm=False); diagH_Ry, P_Ry = ham_Ry.eigh()
    ham_Rz = hamiltonian([["z", [[1, i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rz, P_Rz = ham_Rz.eigh()

    Rx = P_Rx @ np.diag(np.exp(-1j * pulseAngle * diagH_Rx)) @ np.conj(P_Rx).T; Rx_m = P_Rx @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Rx)) @ np.conj(P_Rx).T
    Ry = P_Ry @ np.diag(np.exp(-1j * pulseAngle * diagH_Ry)) @ np.conj(P_Ry).T; Ry_m = P_Ry @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Ry)) @ np.conj(P_Ry).T
    Rz = P_Rz @ np.diag(np.exp(-1j * pulseAngle * diagH_Rz)) @ np.conj(P_Rz).T; Rz_m = P_Rz @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Rz)) @ np.conj(P_Rz).T

    Sz_Tot = SzTot(spinBasis)
    if posMag:
        magMask = np.logical_not(np.isclose(Sz_Tot, 0.0, atol=1e-3)) * (Sz_Tot > 0.0)
        Sz_Tot_comp = Sz_Tot[magMask]
        basisStates_comp = spinBasis.states[magMask]
    else:
        Sz_Tot_comp = Sz_Tot
        basisStates_comp = spinBasis.states

    St_theta_Mat = np.zeros((basisStates_comp.size, tgrid.size), dtype=complex)

    cycleRed = 10; timestep = tgrid[1] - tgrid[0]; T = timestep / cycleRed; tau = T / 6
    U_tau = get_U(diagH, P, tau)
    U_T = Ry @ Ry @ U_tau @ Rx @ U_tau @ Ry_m @ U_tau @ U_tau @ Ry_m @ U_tau @ Rx @ U_tau  # time-evolution for a total cycle time T = dt / cycleRed. Symmetric Heisenberg + x-disorder
    U_dt = np.linalg.matrix_power(U_T, cycleRed)  # time-evolution for a step dt in the time grid

    for indb, bint in enumerate(basisStates_comp):
        bstate = basisVector(bint, spinBasis)

        psi_t_re = np.zeros((bstate.size, tgrid.size), dtype=complex)
        psi_t_im = np.zeros((bstate.size, tgrid.size), dtype=complex)
        U_t = np.eye(U_T.shape[0])

        for indt, t in enumerate(tgrid):
            # U_theta = get_U(diagH, P, t)
            psi_t_re[:, indt] = np.dot(U_t, bstate)
            psi_t_im[:, indt] = np.dot(Rx @ U_t, bstate)
            U_t = U_dt @ U_t

        ave_Sz_Tot_re = aveSzTot(psi_t_re, Sz_Tot, ShotNoise, N_ShotNoise)
        ave_Sz_Tot_im = aveSzTot(psi_t_im, Sz_Tot, ShotNoise, N_ShotNoise)
        ave_Sz_Tot = ave_Sz_Tot_re + 1j * ave_Sz_Tot_im
        St_theta_Mat[indb, :] = ave_Sz_Tot * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

    if posMag:
        St_theta = 2 * np.sum(St_theta_Mat, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector. For some reason St_theta[0] is a little off from the sum rule (N/4) when we compute only over the positive magnetization sector...
    else:
        St_theta = np.sum(St_theta_Mat, axis=0)

    Aw_theta, fVals = spectFunc(St_theta, tgrid, decayRate)
    df = fVals[1] - fVals[0]

    Spectrum_da = xr.DataArray(Aw_theta, coords=[fVals], dims=['f'])
    ResponseFunc_Real_da = xr.DataArray(np.real(St_theta), coords=[tgrid], dims=['t'])
    ResponseFunc_Imag_da = xr.DataArray(np.imag(St_theta), coords=[tgrid], dims=['t'])

    hamiltonian_da = xr.DataArray(hamiltonianListToMatrix(HParams, spinBasis.L, True), coords=[np.arange(spinBasis.L), np.arange(spinBasis.L)], dims=['i', 'j'])

    data_dict = {'Spectrum': Spectrum_da, 'ResponseFunc_Real': ResponseFunc_Real_da, 'ResponseFunc_Imag': ResponseFunc_Imag_da, 'HamiltonianMatrix': hamiltonian_da}
    coords_dict = {'f': fVals, 't': tgrid, 'basisStates_comp': basisStates_comp, 'basisStates_all': spinBasis.states, 'i': np.arange(spinBasis.L), 'j': np.arange(spinBasis.L)}
    attrs_dict = {'Nspins': N}

    trueED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trueED_ds


def trueSim_complex_qT(tgrid, spinBasis, HParams, kappa, gamma_amp, gamma_phase):
    [JijList, hiList] = HParams
    N = spinBasis.L
    operator_list = getSpinOperators(N)

    pulseAngle = np.pi / 2

    Rx, Ry, Rz = makeRotations(operator_list, pulseAngle)
    Rx_m, Ry_m, Rz_m = makeRotations(operator_list, -1*pulseAngle)
    sRx = qt.to_super(Rx); sRy = qt.to_super(Ry); sRz = qt.to_super(Rz)
    sRx_m = qt.to_super(Rx_m); sRy_m = qt.to_super(Ry_m); sRz_m = qt.to_super(Rz_m)

    H, c_op_list = makeGenerator_nmr(operator_list, JijList, hiList, kappa, gamma_amp, gamma_phase)

    Sz_Tot = SzTot(spinBasis)
    magMask = np.logical_not(np.isclose(Sz_Tot, 0.0, atol=1e-3)) * (Sz_Tot > 0.0)
    Sz_Tot_comp = Sz_Tot[magMask]
    basisStates_comp = spinBasis.states[magMask]

    St_theta_Mat = np.zeros((basisStates_comp.size, tgrid.size), dtype=complex)

    dim_list1 = np.ones(N,dtype='int').tolist()
    dim_list2 = (2*np.ones(N,dtype='int')).tolist()

    # ham_Rx = hamiltonian([["x", [[1, i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rx, P_Rx = ham_Rx.eigh()
    # Rx = P_Rx @ np.diag(np.exp(-1j * pulseAngle * diagH_Rx)) @ np.conj(P_Rx).T; Rx_m = P_Rx @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Rx)) @ np.conj(P_Rx).T

    # for indb, bint in enumerate(basisStates_comp):
    #     bstate = basisVector(bint, spinBasis)
    #     # rho0 = qt.Qobj(bstate) * qt.Qobj(bstate).dag()
    #     rho0 = qt.ket2dm(qt.Qobj(bstate,dims=[dim_list2,dim_list1]))
    #     result = qt.mesolve(H, rho0, tgrid, c_op_list)

    #     state_prob_re = np.zeros((bstate.size, tgrid.size), dtype=float)
    #     state_prob_im = np.zeros((bstate.size, tgrid.size), dtype=float)
    #     for indt, t in enumerate(tgrid):
    #         state_prob_re[:, indt] = np.abs(np.diag(result.states[indt].full()))
    #         state_prob_im[:, indt] = np.abs(np.diag(Rx @ result.states[indt].full() @ Rx_m))

    #     ave_Sz_Tot_re = np.sum(np.multiply(Sz_Tot[:, None], state_prob_re), axis=0)
    #     ave_Sz_Tot_im = np.sum(np.multiply(Sz_Tot[:, None], state_prob_im), axis=0)
    #     ave_Sz_Tot = ave_Sz_Tot_re + 1j * ave_Sz_Tot_im

    #     St_theta_Mat[indb, :] = ave_Sz_Tot * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

    L = qt.liouvillian(H,c_op_list)
    dt = tgrid[1] - tgrid[0]
    V_dt = (dt*L).expm()

    for indb, bint in enumerate(basisStates_comp):
        bstate = basisVector(bint, spinBasis)
        rho0 = qt.operator_to_vector(qt.ket2dm(qt.Qobj(bstate,dims=[dim_list2,dim_list1])))
        V_t = qt.to_super(qt.identity(dim_list2))

        state_prob_re = np.zeros((bstate.size, tgrid.size), dtype=float)
        state_prob_im = np.zeros((bstate.size, tgrid.size), dtype=float)
        for indt, t in enumerate(tgrid):
            rho = V_t*rho0
            rho_re = qt.vector_to_operator(rho).full()
            rho_im = qt.vector_to_operator(sRx*rho).full()
            state_prob_re[:, indt] = np.abs(np.diag(rho_re))
            state_prob_im[:, indt] = np.abs(np.diag(rho_im))
            V_t = V_dt*V_t

        ave_Sz_Tot_re = np.sum(np.multiply(Sz_Tot[:, None], state_prob_re), axis=0)
        ave_Sz_Tot_im = np.sum(np.multiply(Sz_Tot[:, None], state_prob_im), axis=0)
        ave_Sz_Tot = ave_Sz_Tot_re + 1j * ave_Sz_Tot_im

        St_theta_Mat[indb, :] = ave_Sz_Tot * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

    St_theta = 2 * np.sum(St_theta_Mat, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector. For some reason St_theta[0] is a little off from the sum rule (N/4) when we compute only over the positive magnetization sector...

    ResponseFunc_Real_da = xr.DataArray(np.real(St_theta), coords=[tgrid], dims=['t'])
    ResponseFunc_Imag_da = xr.DataArray(np.imag(St_theta), coords=[tgrid], dims=['t'])

    data_dict = {'ResponseFunc_Real': ResponseFunc_Real_da, 'ResponseFunc_Imag': ResponseFunc_Imag_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'Nspins': N}

    trueED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trueED_ds

def trueSim_complex_qT_aveHam(tgrid, spinBasis, HParams, kappa, gamma_amp, gamma_phase):
    [JijList, hiList] = HParams
    N = spinBasis.L
    operator_list = getSpinOperators(N)

    pulseAngle = np.pi / 2

    Rx, Ry, Rz = makeRotations(operator_list, pulseAngle)
    Rx_m, Ry_m, Rz_m = makeRotations(operator_list, -1*pulseAngle)
    sRx = qt.to_super(Rx); sRy = qt.to_super(Ry); sRz = qt.to_super(Rz)
    sRx_m = qt.to_super(Rx_m); sRy_m = qt.to_super(Ry_m); sRz_m = qt.to_super(Rz_m)
    H, c_op_list = makeGenerator_sc(operator_list, JijList, hiList, kappa, gamma_amp, gamma_phase)

    Sz_Tot = SzTot(spinBasis)
    magMask = np.logical_not(np.isclose(Sz_Tot, 0.0, atol=1e-3)) * (Sz_Tot > 0.0)
    Sz_Tot_comp = Sz_Tot[magMask]
    basisStates_comp = spinBasis.states[magMask]

    St_theta_Mat = np.zeros((basisStates_comp.size, tgrid.size), dtype=complex)

    dim_list1 = np.ones(N,dtype='int').tolist()
    dim_list2 = (2*np.ones(N,dtype='int')).tolist()

    L = qt.liouvillian(H,c_op_list)
    dt = tgrid[1] - tgrid[0]

    cycleRed = 10; T = dt / cycleRed; tau = T / 6
    V_tau = (tau*L).expm()
    V_T = sRy * sRy * V_tau * sRx * V_tau * sRy_m * V_tau * V_tau * sRy_m * V_tau * sRx * V_tau  # time-evolution for a total cycle time T = dt / cycleRed. Symmetric Heisenberg + x-disorder
    V_dt = V_T**cycleRed

    for indb, bint in enumerate(basisStates_comp):
        bstate = basisVector(bint, spinBasis)
        rho0 = qt.operator_to_vector(qt.ket2dm(qt.Qobj(bstate,dims=[dim_list2,dim_list1])))
        V_t = qt.to_super(qt.identity(dim_list2))

        state_prob_re = np.zeros((bstate.size, tgrid.size), dtype=float)
        state_prob_im = np.zeros((bstate.size, tgrid.size), dtype=float)
        for indt, t in enumerate(tgrid):
            rho = V_t*rho0
            rho_re = qt.vector_to_operator(rho).full()
            rho_im = qt.vector_to_operator(sRx*rho).full()
            state_prob_re[:, indt] = np.abs(np.diag(rho_re))
            state_prob_im[:, indt] = np.abs(np.diag(rho_im))
            V_t = V_dt*V_t

        ave_Sz_Tot_re = np.sum(np.multiply(Sz_Tot[:, None], state_prob_re), axis=0)
        ave_Sz_Tot_im = np.sum(np.multiply(Sz_Tot[:, None], state_prob_im), axis=0)
        ave_Sz_Tot = ave_Sz_Tot_re + 1j * ave_Sz_Tot_im

        St_theta_Mat[indb, :] = ave_Sz_Tot * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

    St_theta = 2 * np.sum(St_theta_Mat, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector. For some reason St_theta[0] is a little off from the sum rule (N/4) when we compute only over the positive magnetization sector...

    ResponseFunc_Real_da = xr.DataArray(np.real(St_theta), coords=[tgrid], dims=['t'])
    ResponseFunc_Imag_da = xr.DataArray(np.imag(St_theta), coords=[tgrid], dims=['t'])

    data_dict = {'ResponseFunc_Real': ResponseFunc_Real_da, 'ResponseFunc_Imag': ResponseFunc_Imag_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'Nspins': N}

    trueED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trueED_ds


def xxzSim_pair_qT(tgrid, J, Delta, hiList, kappa, gamma_amp, gamma_phase):
    N = 2
    operator_list = getSpinOperators(N)
    si_list, sx_list, sy_list, sz_list = operator_list
    sx_tot = sx_list[0] + sx_list[1]; sy_tot = sy_list[0] + sy_list[1]; sz_tot = sz_list[0] + sz_list[1];

    pulseAngle = np.pi / 2

    Rx, Ry, Rz = makeRotations(operator_list, pulseAngle)
    Rx_m, Ry_m, Rz_m = makeRotations(operator_list, -1*pulseAngle)
    sRx = qt.to_super(Rx); sRy = qt.to_super(Ry); sRz = qt.to_super(Rz)
    sRx_m = qt.to_super(Rx_m); sRy_m = qt.to_super(Ry_m); sRz_m = qt.to_super(Rz_m)

    H, c_op_list = makeGenerator_xxz(operator_list, J, Delta, hiList, kappa, gamma_amp, gamma_phase)

    dim_list1 = np.ones(N,dtype='int').tolist()
    dim_list2 = (2*np.ones(N,dtype='int')).tolist()

    L = qt.liouvillian(H,c_op_list)
    dt = tgrid[1] - tgrid[0]
    V_dt = (dt*L).expm()

    rho0_ket = 2*sx_list[1]*qt.tensor(qt.basis(2,0), qt.basis(2,0))
    
    # rho_u = qt.tensor(qt.basis(2,0), qt.basis(2,0))
    # rho_d = 2*sx_list[0]*2*sx_list[1]*rho_u
    # rho_p = (1/np.sqrt(2))*(2*sx_list[1]*rho_u + 2*sx_list[0]*rho_u)
    # rho0_ket = (1/2)*(rho_u - rho_d + 1j*np.sqrt(2)*rho_p)

    rho0 = qt.operator_to_vector(qt.ket2dm(rho0_ket))
    V_t = qt.to_super(qt.identity(dim_list2))

    state_prob = np.zeros((tgrid.size), dtype=float)
    Sy = np.zeros((tgrid.size), dtype=float)
    for indt, t in enumerate(tgrid):
        rho = qt.vector_to_operator(V_t*rho0)
        state_prob[indt] = np.abs(np.diag(rho.full()))[1]
        Sy[indt] = qt.expect(rho,sy_tot)
        V_t = V_dt*V_t

    state_prob_da = xr.DataArray(state_prob, coords=[tgrid], dims=['t'])
    Sy_da = xr.DataArray(Sy, coords=[tgrid], dims=['t'])
    data_dict = {'state_prob': state_prob_da, 'Sy': Sy_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'J': J, 'Delta': Delta, 'kappa': kappa, 'gamma_amp': gamma_amp, 'gamma_phase':gamma_phase}
    ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    
    return ds


def xxzSim_pair_qT_aveHam(tgrid, J_S, J_I, hiList, Delta, alpha, kappa, gamma_amp, gamma_phase):
    N = 2
    operator_list = getSpinOperators(N)
    si_list, sx_list, sy_list, sz_list = operator_list
    sx_tot = sx_list[0] + sx_list[1]; sy_tot = sy_list[0] + sy_list[1]; sz_tot = sz_list[0] + sz_list[1];

    pulseAngle = np.pi / 2

    Rx, Ry, Rz = makeRotations(operator_list, pulseAngle)
    Rx_m, Ry_m, Rz_m = makeRotations(operator_list, -1*pulseAngle)
    sRx = qt.to_super(Rx); sRy = qt.to_super(Ry); sRz = qt.to_super(Rz)
    sRx_m = qt.to_super(Rx_m); sRy_m = qt.to_super(Ry_m); sRz_m = qt.to_super(Rz_m)

    H, c_op_list = makeGenerator_native(operator_list, J_S, J_I, hiList, kappa, gamma_amp, gamma_phase)

    dim_list1 = np.ones(N,dtype='int').tolist()
    dim_list2 = (2*np.ones(N,dtype='int')).tolist()

    L = qt.liouvillian(H,c_op_list)
    dt = tgrid[1] - tgrid[0]

    # V_dt = (dt*L).expm()

    alpha = (J_I * Delta + J_S * (Delta - 2))/(J_I - J_S * Delta)  # pulse anisotropy parameter
    cycleRed = 10; T = dt / cycleRed; tau = T / (4+2*alpha)
    V_tau = (tau*L).expm()
    V_tau_z = (alpha*tau*L).expm()
    V_T = sRz * sRz * V_tau_z * sRx_m * V_tau * sRy * V_tau * sRy * sRy * V_tau * sRy_m * V_tau * sRx * V_tau_z  # time-evolution for a total cycle time T = dt / cycleRed. Symmetric Heisenberg + x-disorder
    V_dt = V_T**cycleRed

    rho0_ket = 2*sx_list[1]*qt.tensor(qt.basis(2,0), qt.basis(2,0))
    
    # rho_u = qt.tensor(qt.basis(2,0), qt.basis(2,0))
    # rho_d = 2*sx_list[0]*2*sx_list[1]*rho_u
    # rho_p = (1/np.sqrt(2))*(2*sx_list[1]*rho_u + 2*sx_list[0]*rho_u)
    # rho0_ket = (1/2)*(rho_u - rho_d + 1j*np.sqrt(2)*rho_p)

    rho0 = qt.operator_to_vector(qt.ket2dm(rho0_ket))
    V_t = qt.to_super(qt.identity(dim_list2))

    state_prob = np.zeros((tgrid.size), dtype=float)
    Sy = np.zeros((tgrid.size), dtype=float)
    for indt, t in enumerate(tgrid):
        rho = qt.vector_to_operator(V_t*rho0)
        state_prob[indt] = np.abs(np.diag(rho.full()))[1]
        Sy[indt] = qt.expect(rho,sy_tot)
        V_t = V_dt*V_t

    state_prob_da = xr.DataArray(state_prob, coords=[tgrid], dims=['t'])
    Sy_da = xr.DataArray(Sy, coords=[tgrid], dims=['t'])
    data_dict = {'state_prob': state_prob_da, 'Sy': Sy_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'J_S': J_S, 'J_I': J_I, 'Delta': Delta, 'kappa': kappa, 'gamma_amp': gamma_amp, 'gamma_phase':gamma_phase}
    ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    
    return ds