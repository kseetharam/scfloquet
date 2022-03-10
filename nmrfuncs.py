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

def matDistance(M1, M2):
    # takes two Hamiltonian matrices and computes the square distance between the upper half of the parameters (=unique values in the matrix)
    diffList = []
    for i in np.arange(M1.shape[0]):
        for j in np.arange(M1.shape[0]):
            if (i < j) or (i == j):
                diffList.append((M1[i, j] - M2[i, j])**2)
    return np.sum(np.array(diffList))


def get_GateCount(trotStep, Nspins, N_nonzeroJij):
    # returns total number of gates and experimental time (in seconds) for single sample taken
    # assume that we have combined the Jij and Jji gates into one rotation; if we do them separately then num_IntGates -> 2*num_IntGates below
    # N_nonzeroJij is the number of unique nonzero interaction parameters Jij. This may be less than binom(Nspins, 2).
    N_zeroJij = binom(Nspins, 2) - N_nonzeroJij
    # print(N_zeroJij)
    num_IntGates = int(3 * N_nonzeroJij * trotStep)
    num_CShiftGates = 1 * trotStep
    ExpTime = num_IntGates * (125 * 1e-6) + num_CShiftGates * (15 * 1e-6)
    return (num_IntGates + num_CShiftGates), ExpTime


def XXgateFidelity(expTime, trotStep, tMax):
    # takes experimental time (in seconds) and largest rotation angle of gate (= tMax/trotStep) and returns
    theta = tMax / trotStep
    return 0.5 + 0.5 * 1 / np.sqrt(1 + ((2 * theta)**2) * ((0.1 * expTime * 1e3)**2))


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


def stateFidelity(psi1, psi2):
    return np.sqrt(np.abs(np.dot(psi1, psi2))**2)


def hellingerDistance(A1, A2, fVals):
    A1mask = A1 < 0; A2mask = A2 < 0
    if np.any(A1mask) or np.any(A2mask):
        # print('Neg Spectrum')
        # print('\n'); print(A1); print('\n'); print(A2)
        A1[A1mask] = 0; A2[A2mask] = 0

    # integrand = (np.sqrt(A1) - np.sqrt(A2))**2

    A1_norm = A1 / simps(y=A1, x=fVals)
    A2_norm = A2 / simps(y=A2, x=fVals)
    integrand = (np.sqrt(A1_norm) - np.sqrt(A2_norm))**2

    return np.sqrt(0.5 * simps(y=integrand, x=fVals))


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

# ---- ION HEATING NOISE FUNCTIONS ----


def generateGateAngle(theta_input, c2, gateTime, csAmplitude):
    csMean = np.array([np.real(csAmplitude), np.imag(csAmplitude)])
    csCov = (c2 * gateTime / 2) * np.eye(2)
    posteriorRV = multivariate_normal(mean=csMean, cov=csCov)
    new_csSample = posteriorRV.rvs(size=1)
    new_csAmplitude = new_csSample[0] + 1j * new_csSample[1]
    theta = theta_input * np.exp(-1 * np.abs(new_csAmplitude)**2)
    return theta, new_csAmplitude


def generateTimeShift(dt, Jij, c2, currentExpTime, gateTime, csAmplitude, heatingCorrection, theta_input):
    if np.isclose(c2, 0):
        return 0, csAmplitude
    if np.isclose(currentExpTime, 0):
        return 0, csAmplitude
    theta_perfect = Jij * dt / 4
    # if heatingCorrection and (c2 * currentExpTime <= 1.0):
    #     theta_input = theta_perfect * (1 + c2 * currentExpTime)  # This corrects the average applied angle
    if heatingCorrection:
        # print(c2 * currentExpTime, theta_perfect / np.pi)  # seems we reach a max of max(c2t) ~ 1.25 and max(theta_perfect) ~ 0.7*Pi
        # input_angle = optimumInputAngle(c2, currentExpTime, theta_perfect)  # This optimizes the average fidelity
        input_angle = theta_input
    else:
        input_angle = theta_perfect
    thetaSample, new_csAmplitude = generateGateAngle(input_angle, c2, gateTime, csAmplitude)
    dt_effective = 4 * thetaSample / Jij
    dt_err = dt_effective - dt
    return dt_err, new_csAmplitude


def optimumInputAngle(c2, currentExpTime, theta_perfect):
    optrange = [1e-5, 2 * np.pi]
    # optrange = [theta_perfect, 1.5 * theta_perfect]
    def fopt(theta_input): return averageGateFidelity(theta_input, c2, currentExpTime, theta_perfect) - gateFidelity(theta_input, theta_perfect)
    tpsign = np.sign(theta_perfect)
    try:
        sol = root_scalar(fopt, bracket=[tpsign * optrange[0], tpsign * optrange[1]], method='brentq')
        theta_input_opt = sol.root
        finite_fidelity = averageGateFidelity(theta_input_opt, c2, currentExpTime, theta_perfect)
        zero_fidelity = averageGateFidelity(0, c2, currentExpTime, theta_perfect)
        inf_fidelity = averageGateFidelity(1e10, c2, currentExpTime, theta_perfect)
        fidOptions = np.array([finite_fidelity, zero_fidelity, inf_fidelity])
        angleOptions = np.array([theta_input_opt, 0, 1e10])
        maxInd = np.argmax(fidOptions)
        outputAngle = angleOptions[maxInd]
    except ValueError:
        print('Root Scalar Value Error')
        outputAngle = optimumInputAngle_grid(c2, currentExpTime, theta_perfect)
        print(theta_perfect / np.pi, outputAngle / np.pi)
    except Exception as e:
        print('Root Scalar Strange Error')
        outputAngle = optimumInputAngle_grid(c2, currentExpTime, theta_perfect)
        print(theta_perfect / np.pi, outputAngle / np.pi)
    return outputAngle


def optimumInputAngle_grid(c2, currentExpTime, theta_perfect):
    theta_input_Vals = np.linspace(theta_perfect, 1.5 * theta_perfect, 1000)
    fopt_Vals = averageGateFidelity_vectorized(theta_input_Vals, c2, currentExpTime, theta_perfect) - gateFidelity(theta_input_Vals, theta_perfect)
    solind = np.argmin(np.abs(fopt_Vals))
    theta_input_opt = theta_input_Vals[solind]
    finite_fidelity = averageGateFidelity(theta_input_opt, c2, currentExpTime, theta_perfect)
    zero_fidelity = averageGateFidelity(0, c2, currentExpTime, theta_perfect)
    inf_fidelity = averageGateFidelity(1e10, c2, currentExpTime, theta_perfect)
    fidOptions = np.array([finite_fidelity, zero_fidelity, inf_fidelity])
    angleOptions = np.array([theta_input_opt, 0, 1e10])
    maxInd = np.argmax(fidOptions)
    # if maxInd == 2:
    #     print(c2, theta_perfect / np.pi, theta_input_opt / np.pi, theta_input_opt / theta_perfect, finite_fidelity)
    return angleOptions[maxInd]


def averageGateFidelity(theta_input, c2, currentExpTime, theta_perfect):
    eta = c2 * currentExpTime
    hyp1f2 = mpm.hyp1f2
    return (1 / 2) + (1 / 2) * np.cos(theta_input / 2) * np.cos(theta_perfect / 2) + (eta * (theta_input**2) / (8 + 16 * eta)) * np.cos(theta_perfect / 2) * hyp1f2(1 + 1 / (2 * eta), 3 / 2, 2 + 1 / (2 * eta), -(theta_input**2) / 16) + (theta_input / (4 + 4 * eta)) * np.sin(theta_perfect / 2) * hyp1f2(1 / 2 + 1 / (2 * eta), 3 / 2, 3 / 2 + 1 / (2 * eta), -(theta_input**2) / 16)


def averageGateFidelity_vectorized(theta_input, c2, currentExpTime, theta_perfect):
    eta = c2 * currentExpTime
    hyp1f2 = np.vectorize(mpm.hyp1f2)
    return (1 / 2) + (1 / 2) * np.cos(theta_input / 2) * np.cos(theta_perfect / 2) + (eta * (theta_input**2) / (8 + 16 * eta)) * np.cos(theta_perfect / 2) * hyp1f2(1 + 1 / (2 * eta), 3 / 2, 2 + 1 / (2 * eta), -(theta_input**2) / 16) + (theta_input / (4 + 4 * eta)) * np.sin(theta_perfect / 2) * hyp1f2(1 / 2 + 1 / (2 * eta), 3 / 2, 3 / 2 + 1 / (2 * eta), -(theta_input**2) / 16)


def gateFidelity(theta_input, theta_perfect):
    return np.cos((theta_input - theta_perfect) / 4)**2


# def averageGateFidelity_cos(theta_input, c2, currentExpTime, theta_perfect):
#     eta = c2 * currentExpTime
#     hyp1f2 = mpm.hyp1f2
#     return np.cos(theta_perfect / 4) * hyp1f2(1 / (2 * eta), 1 / 2, 1 + 1 / (2 * eta), -(theta_input**2) / 64) + np.sin(theta_perfect / 4) * (theta_input / (4 + 4 * eta)) * hyp1f2(1 / 2 + 1 / (2 * eta), 3 / 2, 3 / 2 + 1 / (2 * eta), -(theta_input**2) / 64)


# def averageGateFidelity_cos_vectorized(theta_input, c2, currentExpTime, theta_perfect):
#     eta = c2 * currentExpTime
#     hyp1f2 = np.vectorize(mpm.hyp1f2)
#     return np.cos(theta_perfect / 4) * hyp1f2(1 / (2 * eta), 1 / 2, 1 + 1 / (2 * eta), -(theta_input**2) / 64) + np.sin(theta_perfect / 4) * (theta_input / (4 + 4 * eta)) * hyp1f2(1 / 2 + 1 / (2 * eta), 3 / 2, 3 / 2 + 1 / (2 * eta), -(theta_input**2) / 64)


# def gateFidelity_cos(theta_input, theta_perfect):
#     return np.cos((theta_input - theta_perfect) / 4)

# ---- ED FUNCTIONS ----


def fullHamiltonian(spinBasis, JijList, hiList, chemicalShifts=True):
    # takes parameters Jij and hi (in Hz) and outputs the Hamiltonian
    # chemicalShift is a flag that decides whether we include the hi terms or just the interactions Jij
    static = [["xx", JijList], ["yy", JijList], ["zz", JijList]]; dynamic = []
    # static = [["xx", JijList]]; dynamic = []; print('zz')
    if chemicalShifts:
        static.append(["x", hiList])
    return hamiltonian(static, dynamic, basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False)


def XYZHamiltonian(spinBasis, JijList, hiList, chemicalShifts=True):
    static = [["xx", JijList]]; dynamic = []
    JijyList = []
    JijzList = []

    yfac = 1.0
    # zfac = 0.8
    zfac = 4.3

    # yfac = 1.1
    # zfac = 1.0

    for tup in JijList:
        Jij, i, j = tup
        JijyList.append([yfac * Jij, i, j])
        JijzList.append([zfac * Jij, i, j])
    static.append(["yy", JijyList])
    static.append(["zz", JijzList])
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


def makeGenerators(JijList, hiList, N, gamma, decohType):
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

    H = 0

    for tup in hiList:
        hi, i = tup
        if hi != 0:
            H += hi * sx_list[i]

    for tup in JijList:
        Jij, i, j = tup
        H += Jij * (sx_list[i] * sx_list[j] + sy_list[i] * sy_list[j] + sz_list[i] * sz_list[j])

    c_op_list = []

    if gamma > 0.0:
        if decohType == 'individual':
            print('individual dissipation')
            for i in range(N):
                c_op_list.append(np.sqrt(1 - gamma) * si_list[i])
                c_op_list.append(np.sqrt(gamma / 3) * sx_list[i])
                c_op_list.append(np.sqrt(gamma / 3) * sy_list[i])
                c_op_list.append(np.sqrt(gamma / 3) * sz_list[i])
        elif decohType == 'symmetric':
            print('symmetric dissipation')
            c_op_list.append(np.sqrt(1 - 3 * gamma) * (si_list[0] + si_list[1] + si_list[2]))
            c_op_list.append(np.sqrt(gamma) * (sx_list[0] + sx_list[1] + sx_list[2])); c_op_list.append(np.sqrt(gamma) * (sy_list[0] + sy_list[1] + sy_list[2])); c_op_list.append(np.sqrt(gamma) * (sz_list[0] + sz_list[1] + sz_list[2]))
            c_op_list.append(np.sqrt(1 - gamma) * si_list[3])
            c_op_list.append(np.sqrt(gamma / 3) * sx_list[3]); c_op_list.append(np.sqrt(gamma / 3) * sy_list[3]); c_op_list.append(np.sqrt(gamma / 3) * sz_list[3])
        else:
            print('Decoherence Error')

    return H, c_op_list


def gateHamiltonians(spinBasis, JijList, hiList, chemicalShifts=True):
    # takes parameters Jij and hi (in Hz) and outputs the Hamiltonians corresponding to the list of xx, yy, zz, and x gates (for each pair of spins)
    # chemicalShift is a flag that decides whether we include the x gate (corresponding to the hi terms) or just the xx, yy, and zz gates (corresponding to interactions Jij)
    HxxList = [hamiltonian([["xx", [Jij]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False) for Jij in JijList]
    HyyList = [hamiltonian([["yy", [Jij]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False) for Jij in JijList]
    HzzList = [hamiltonian([["zz", [Jij]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False) for Jij in JijList]
    JijVals = [Jij[0] for Jij in JijList]
    Hx = hamiltonian([["x", hiList]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False)
    hamList = [HxxList, HyyList, HzzList]
    if chemicalShifts:
        hamList.append(Hx)
    return hamList, JijVals


def trotter_equal(psi0, t, gateHamiltonianList_diag, JijVals, trotParamDict, theta_input_Vals):
    # equal Trotter step for all interactions
    # note: trotStep should be an integer
    if np.isclose(t, 0):
        return psi0, 0
    [HxxList_diag, HyyList_diag, HzzList_diag, Hx_diag] = gateHamiltonianList_diag
    JijArray = np.array(JijVals)

    gateBudget = trotParamDict['gateCount']; c2 = trotParamDict['c2']; intGateTime = trotParamDict['intGateTime']; singRotTime = trotParamDict['singRotTime']; eqTrot = trotParamDict['eqTrot']
    heatingCorrection = trotParamDict['heatingCorrection']
    Nspins = trotParamDict['N']
    # trotStep = int(gateBudget / (3 * len(JijVals) + 2 * Nspins))  # Number of interaction gates = 3*len(JijVals); Number of single qubit rotations = 2*Nspins
    trotStep = int(gateBudget / (3 * len(JijVals)))  # Number of interaction gates = 3*len(JijVals); Number of single qubit rotations = 2*Nspins
    # print('Trotter Step: {0}'.format(gateBudget / trotStep))
    # trotStep = int(gateBudget / (3 * len(JijVals)))  # Number of interaction gates = 3*len(JijVals); Number of single qubit rotations = 2*Nspins
    dt = t / trotStep

    psi_temp = psi0
    csAmplitude = 0 + 0j  # Should technically pull this from a thermal distribution with a temperature set by how much they cool the ions between each shot
    counter = 0
    currentExpTime = 0
    # gateCounter_xx = np.zeros(JijArray.size); gateCounter_yy = np.zeros(JijArray.size); gateCounter_zz = np.zeros(JijArray.size)

    tauMax = trotStep * 3 * len(JijVals) * intGateTime + 2 * Nspins * singRotTime  # Latest experimental time
    # print('c2*tau_Max={0}'.format(c2 * tauMax))
    # print(trotStep)

    for ind in np.arange(trotStep):
        for indHx, Hxx_pair in enumerate(HxxList_diag):
            Jij = JijArray[indHx]
            dt_err, csAmplitude = generateTimeShift(dt, Jij, c2, currentExpTime, intGateTime, csAmplitude, heatingCorrection, theta_input_Vals[counter])
            diagH, P = Hxx_pair
            Uxx = get_U(diagH, P, dt + dt_err)
            psi_temp = np.dot(Uxx, psi_temp)
            currentExpTime += intGateTime
            counter += 1
            # gateCounter_xx[indHx] += 1
        for indHy, Hyy_pair in enumerate(HyyList_diag):
            Jij = JijArray[indHy]
            dt_err, csAmplitude = generateTimeShift(dt, Jij, c2, currentExpTime, intGateTime, csAmplitude, heatingCorrection, theta_input_Vals[counter])
            diagH, P = Hyy_pair
            Uyy = get_U(diagH, P, dt + dt_err)
            psi_temp = np.dot(Uyy, psi_temp)
            currentExpTime += intGateTime
            counter += 1
            # gateCounter_yy[indHy] += 1
        currentExpTime += Nspins * singRotTime  # this corresponds to a pi/2 rotation in y to make a zz gate
        for indHz, Hzz_pair in enumerate(HzzList_diag):
            Jij = JijArray[indHz]
            dt_err, csAmplitude = generateTimeShift(dt, Jij, c2, currentExpTime, intGateTime, csAmplitude, heatingCorrection, theta_input_Vals[counter])
            diagH, P = Hzz_pair
            Uzz = get_U(diagH, P, dt + dt_err)
            psi_temp = np.dot(Uzz, psi_temp)
            currentExpTime += intGateTime
            counter += 1
            # gateCounter_zz[indHz] += 1
        diagH, P = Hx_diag[0]
        Ux = get_U(diagH, P, dt)
        psi_temp = np.dot(Ux, psi_temp)
        # currentExpTime += singRotTime  # Technically this is done as a phase advance in software and shouldn't take any time?
        currentExpTime += Nspins * singRotTime  # this corresponds to a -pi/2 rotation in y to undo the previous rotation
    # gateTimes_xx = gateCounter_xx * dt; gateTimes_yy = gateCounter_yy * dt; gateTimes_zz = gateCounter_zz * dt
    # print(t)
    # print(np.allclose(np.abs(gateTimes_xx - t), 0), np.allclose(np.abs(gateTimes_yy - t), 0), np.allclose(np.abs(gateTimes_zz - t), 0))

    return psi_temp, counter


def inputAngle_precompute(t, gateHamiltonianList_diag, JijVals, trotParamDict):
    # equal Trotter step for all interactions
    # note: trotStep should be an integer
    [HxxList_diag, HyyList_diag, HzzList_diag, Hx_diag] = gateHamiltonianList_diag
    JijArray = np.array(JijVals)

    gateBudget = trotParamDict['gateCount']; c2 = trotParamDict['c2']; intGateTime = trotParamDict['intGateTime']; singRotTime = trotParamDict['singRotTime']; eqTrot = trotParamDict['eqTrot']
    heatingCorrection = trotParamDict['heatingCorrection']
    Nspins = trotParamDict['N']
    # trotStep = int(gateBudget / (3 * len(JijVals) + 2 * Nspins))  # Number of interaction gates = 3*len(JijVals); Number of single qubit rotations = 2*Nspins
    trotStep = int(gateBudget / (3 * len(JijVals)))  # Number of interaction gates = 3*len(JijVals); Number of single qubit rotations = 2*Nspins
    # print('Trotter Step: {0}'.format(gateBudget / trotStep))
    # trotStep = int(gateBudget / (3 * len(JijVals)))  # Number of interaction gates = 3*len(JijVals); Number of single qubit rotations = 2*Nspins
    dt = t / trotStep

    counter = 0
    currentExpTime = 0
    # gateCounter_xx = np.zeros(JijArray.size); gateCounter_yy = np.zeros(JijArray.size); gateCounter_zz = np.zeros(JijArray.size)

    tauMax = trotStep * 3 * len(JijVals) * intGateTime + 2 * Nspins * singRotTime  # Latest experimental time
    # print('c2*tau_Max={0}'.format(c2 * tauMax))

    theta_input_Vals = np.zeros(trotStep * 3 * len(HxxList_diag))

    for ind in np.arange(trotStep):
        for indHx, Hxx_pair in enumerate(HxxList_diag):
            Jij = JijArray[indHx]
            if np.isclose(currentExpTime, 0):
                theta_input_Vals[counter] = Jij * dt / 4
            else:
                theta_input_Vals[counter] = optimumInputAngle(c2, currentExpTime, Jij * dt / 4)
            currentExpTime += intGateTime
            counter += 1
            # gateCounter_xx[indHx] += 1
        for indHy, Hyy_pair in enumerate(HyyList_diag):
            Jij = JijArray[indHy]
            theta_input_Vals[counter] = optimumInputAngle(c2, currentExpTime, Jij * dt / 4)
            currentExpTime += intGateTime
            counter += 1
            # gateCounter_yy[indHy] += 1
        currentExpTime += Nspins * singRotTime  # this corresponds to a pi/2 rotation in y to make a zz gate
        for indHz, Hzz_pair in enumerate(HzzList_diag):
            Jij = JijArray[indHz]
            theta_input_Vals[counter] = optimumInputAngle(c2, currentExpTime, Jij * dt / 4)
            currentExpTime += intGateTime
            counter += 1
            # gateCounter_zz[indHz] += 1
        # currentExpTime += singRotTime  # Technically this is done as a phase advance in software and shouldn't take any time?
        currentExpTime += Nspins * singRotTime  # this corresponds to a -pi/2 rotation in y to undo the previous rotation
    # gateTimes_xx = gateCounter_xx * dt; gateTimes_yy = gateCounter_yy * dt; gateTimes_zz = gateCounter_zz * dt
    # print(t)
    # print(np.allclose(np.abs(gateTimes_xx - t), 0), np.allclose(np.abs(gateTimes_yy - t), 0), np.allclose(np.abs(gateTimes_zz - t), 0))
    return theta_input_Vals


def diagHamList(gateHamiltonianList):
    [HxxList, HyyList, HzzList, Hx] = gateHamiltonianList
    HxxList_diag = []; HyyList_diag = []; HzzList_diag = []; Hx_diag = []
    for indx, Hxx in enumerate(HxxList):
        HxxList_diag.append(Hxx.eigh())
    for indy, Hyy in enumerate(HyyList):
        HyyList_diag.append(Hyy.eigh())
    for indz, Hzz in enumerate(HzzList):
        HzzList_diag.append(Hzz.eigh())
    Hx_diag.append(Hx.eigh())
    gateHamiltonianList_diag = [HxxList_diag, HyyList_diag, HzzList_diag, Hx_diag]
    return gateHamiltonianList_diag


# def simComp_old(sim_ds1, sim_ds2):
#     # compares the spectrum and average state fideity between two protocol simulations
#     # (assumes one of the simulation is the 'true' result)
#     # Assumes that fVals1 and fVals2 are the same. Assumes that basisStates_PosMag is the same for both datasets
#     Aw1 = sim_ds1['Spectrum'].values
#     fVals1 = sim_ds1.coords['f'].values
#     Aw2 = sim_ds2['Spectrum'].values
#     # fVals2 = sim_ds2.coords['f'].values
#     posmask = fVals1 >= 0
#     Aw1_pos = Aw1[posmask]
#     Aw2_pos = Aw2[posmask]
#     Aw1_mask = Aw1_pos < 0
#     Aw2_mask = Aw2_pos < 0
#     Aw1_pos[Aw1_mask] = 0; Aw1_pos[Aw2_mask] = 0
#     Aw2_pos[Aw1_mask] = 0; Aw2_pos[Aw2_mask] = 0
#     hellDist = hellingerDistance(Aw1_pos, Aw2_pos, fVals1[posmask])

#     basisStates_PosMag = sim_ds1.coords['basisStates_PosMag'].values
#     fidelityArray = np.zeros(basisStates_PosMag.size)
#     for indb, bint in enumerate(basisStates_PosMag):
#         psi_final1 = sim_ds1.sel(basisStates_PosMag=bint)['psiFinal_Real'].values + 1j * sim_ds1.sel(basisStates_PosMag=bint)['psiFinal_Imag'].values
#         psi_final2 = sim_ds2.sel(basisStates_PosMag=bint)['psiFinal_Real'].values + 1j * sim_ds2.sel(basisStates_PosMag=bint)['psiFinal_Imag'].values
#         fidelityArray[indb] = stateFidelity(psi_final1, psi_final2)
#     aveStateFidelity = np.nanmean(fidelityArray)

#     return hellDist, aveStateFidelity

def simComp(sim_ds1, sim_ds2, saveAllStates=False):
    # compares the spectrum and average state fideity between two protocol simulations
    # (assumes one of the simulation is the 'true' result)
    # Assumes that fVals1 and fVals2 are the same. Assumes that basisStates_PosMag is the same for both datasets
    Aw1 = sim_ds1['Spectrum'].values
    fVals1 = sim_ds1.coords['f'].values
    Aw2 = sim_ds2['Spectrum'].values
    # fVals2 = sim_ds2.coords['f'].values
    posmask = fVals1 >= 0
    Aw1_pos = Aw1[posmask]
    Aw2_pos = Aw2[posmask]
    Aw1_mask = Aw1_pos < 0
    Aw2_mask = Aw2_pos < 0
    Aw1_pos[Aw1_mask] = 0; Aw1_pos[Aw2_mask] = 0
    Aw2_pos[Aw1_mask] = 0; Aw2_pos[Aw2_mask] = 0
    hellDist = hellingerDistance(Aw1_pos, Aw2_pos, fVals1[posmask])

    basisStates_comp = sim_ds1.coords['basisStates_comp'].values

    if saveAllStates:
        tgrid = sim_ds1.coords['t'].values
        aveStateFidelity = np.zeros(tgrid.size)
        for indt, t in enumerate(tgrid):
            overlapArray = np.zeros(basisStates_comp.size, dtype='complex')
            for indb, bint in enumerate(basisStates_comp):
                psi_final1 = sim_ds1.sel(basisStates_comp=bint)['psiFinal_Real'].values + 1j * sim_ds1.sel(basisStates_comp=bint)['psiFinal_Imag'].values
                psi_final2 = sim_ds2.sel(basisStates_comp=bint)['psiFinal_Real'].values + 1j * sim_ds2.sel(basisStates_comp=bint)['psiFinal_Imag'].values
                overlapArray[indb] = np.dot(np.conj(psi_final1[:, indt]), psi_final2[:, indt])
            aveStateFidelity[indt] = np.real(np.abs(np.nanmean(overlapArray))**2)
            # aveStateFidelity[indt] = np.real(np.nanmean(np.abs(overlapArray)**2))
    else:
        overlapArray = np.zeros(basisStates_comp.size, dtype='complex')
        for indb, bint in enumerate(basisStates_comp):
            psi_final1 = sim_ds1.sel(basisStates_comp=bint)['psiFinal_Real'].values + 1j * sim_ds1.sel(basisStates_comp=bint)['psiFinal_Imag'].values
            psi_final2 = sim_ds2.sel(basisStates_comp=bint)['psiFinal_Real'].values + 1j * sim_ds2.sel(basisStates_comp=bint)['psiFinal_Imag'].values
            overlapArray[indb] = np.dot(np.conj(psi_final1), psi_final2)
        aveStateFidelity = np.real(np.abs(np.nanmean(overlapArray))**2)
        # aveStateFidelity = np.real(np.nanmean(np.abs(overlapArray)**2))

    return hellDist, aveStateFidelity


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


def trueSim_cosy(tgrid, spinBasis, HParams, shotNoiseParams, decayRate, posMag=True):
    tgrid1, tgrid2 = tgrid; tsize = tgrid1.size * tgrid2.size
    ShotNoise = shotNoiseParams['ShotNoise']; N_ShotNoise = shotNoiseParams['N_ShotNoise']
    [JijList, hiList] = HParams
    H_theta = fullHamiltonian(spinBasis, JijList, hiList, chemicalShifts=True)
    diagH, P = H_theta.eigh()

    N = spinBasis.L

    pulseAngle = np.pi / 2

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

    St_theta_Mat = np.zeros((basisStates_comp.size, tsize), dtype=complex)
    print(basisStates_comp.size)

    for indb, bint in enumerate(basisStates_comp):
        bstate = basisVector(bint, spinBasis)

        psi_t_re = np.zeros((bstate.size, tsize), dtype=complex)
        psi_t_im = np.zeros((bstate.size, tsize), dtype=complex)
        indt = 0
        # tpairList = []
        # tcheckList = []
        for ind2, t2 in enumerate(tgrid2):
            for ind1, t1 in enumerate(tgrid1):
                U_t1 = get_U(diagH, P, t1)
                U_t2 = get_U(diagH, P, t2)

                psi_t_re[:, indt] = np.dot(U_t2 @ U_Rz_m @ U_Rx_m @ U_t1, bstate)
                psi_t_im[:, indt] = np.dot(U_t2 @ U_Ry @ U_t1, bstate)

                # psi_t_re[:, indt] = np.dot(U_t2 @ U_Rz_m @ U_t1 @ U_Rx_m, bstate)
                # psi_t_im[:, indt] = np.dot(U_Rx @ U_t2 @ U_Rz_m @ U_t1 @ U_Rx_m, bstate)

                # psi_t_re[:, indt] = np.dot(U_t2 @ U_Rz_m @ U_t1, bstate)
                # psi_t_im[:, indt] = np.dot(U_Rx @ U_t2 @ U_Rz_m @ U_t1, bstate)

                # tpairList.append((t1, t2))
                indt += 1
                # tcheckList.append(ind1 * ind2)

        ave_Sz_Tot_re = aveSzTot(psi_t_re, Sz_Tot, ShotNoise, N_ShotNoise)
        ave_Sz_Tot_im = aveSzTot(psi_t_im, Sz_Tot, ShotNoise, N_ShotNoise)
        ave_Sz_Tot = ave_Sz_Tot_re + 1j * ave_Sz_Tot_im
        St_theta_Mat[indb, :] = ave_Sz_Tot * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

    if posMag:
        St_theta = 2 * np.sum(St_theta_Mat, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector. For some reason St_theta[0] is a little off from the sum rule (N/4) when we compute only over the positive magnetization sector...
    else:
        St_theta = np.sum(St_theta_Mat, axis=0)

    St_theta = St_theta.reshape((tgrid2.size, tgrid1.size))  # rows of St_theta indexed by t2 and columns indexed by t1 in line with Spinach convention

    ResponseFunc_Real_da = xr.DataArray(np.real(St_theta), coords=[tgrid2, tgrid1], dims=['t2', 't1'])
    ResponseFunc_Imag_da = xr.DataArray(np.imag(St_theta), coords=[tgrid2, tgrid1], dims=['t2', 't1'])

    data_dict = {'ResponseFunc_Real': ResponseFunc_Real_da, 'ResponseFunc_Imag': ResponseFunc_Imag_da}
    coords_dict = {'t1': tgrid1, 't2': tgrid2, 'basisStates_comp': basisStates_comp, 'basisStates_all': spinBasis.states, 'i': np.arange(spinBasis.L), 'j': np.arange(spinBasis.L)}
    attrs_dict = {'Nspins': N}

    trueED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trueED_ds


def trueSim_cosy_aveHam(tgrid, spinBasis, HParams, shotNoiseParams, decayRate, posMag=True):
    tgrid1, tgrid2 = tgrid; tsize = tgrid1.size * tgrid2.size
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

    St_theta_Mat = np.zeros((basisStates_comp.size, tsize), dtype=complex)
    print(basisStates_comp.size)

    cycleRed = 10; timestep = tgrid1[1] - tgrid1[0]; T = timestep / cycleRed; tau = T / 6
    U_tau = get_U(diagH, P, tau)
    U_T = Ry @ Ry @ U_tau @ Rx @ U_tau @ Ry_m @ U_tau @ U_tau @ Ry_m @ U_tau @ Rx @ U_tau  # time-evolution for a total cycle time T = dt / cycleRed. Symmetric Heisenberg + x-disorder
    U_dt = np.linalg.matrix_power(U_T, cycleRed)  # time-evolution for a step dt in the time grid

    for indb, bint in enumerate(basisStates_comp):
        bstate = basisVector(bint, spinBasis)

        psi_t_re = np.zeros((bstate.size, tsize), dtype=complex)
        psi_t_im = np.zeros((bstate.size, tsize), dtype=complex)
        indt = 0
        # tpairList = []
        # tcheckList = []

        U_t2 = np.eye(U_T.shape[0])
        for ind2, t2 in enumerate(tgrid2):
            U_t1 = np.eye(U_T.shape[0])
            for ind1, t1 in enumerate(tgrid1):
                # U_t1 = get_U(diagH, P, t1)
                # U_t2 = get_U(diagH, P, t2)

                # U_t1 = np.linalg.matrix_power(U_dt, ind1)
                # U_t2 = np.linalg.matrix_power(U_dt, ind2)

                psi_t_re[:, indt] = np.dot(U_t2 @ Rz_m @ Rx_m @ U_t1, bstate)
                psi_t_im[:, indt] = np.dot(U_t2 @ Ry @ U_t1, bstate)

                # psi_t_re[:, indt] = np.dot(U_t2 @ U_Rz_m @ U_t1 @ U_Rx_m, bstate)
                # psi_t_im[:, indt] = np.dot(U_Rx @ U_t2 @ U_Rz_m @ U_t1 @ U_Rx_m, bstate)

                # psi_t_re[:, indt] = np.dot(U_t2 @ U_Rz_m @ U_t1, bstate)
                # psi_t_im[:, indt] = np.dot(U_Rx @ U_t2 @ U_Rz_m @ U_t1, bstate)

                # tpairList.append((t1, t2))
                indt += 1
                # tcheckList.append(ind1 * ind2)

                U_t1 = U_dt @ U_t1
            U_t2 = U_dt @ U_t2

        ave_Sz_Tot_re = aveSzTot(psi_t_re, Sz_Tot, ShotNoise, N_ShotNoise)
        ave_Sz_Tot_im = aveSzTot(psi_t_im, Sz_Tot, ShotNoise, N_ShotNoise)
        ave_Sz_Tot = ave_Sz_Tot_re + 1j * ave_Sz_Tot_im
        St_theta_Mat[indb, :] = ave_Sz_Tot * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

    if posMag:
        St_theta = 2 * np.sum(St_theta_Mat, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector. For some reason St_theta[0] is a little off from the sum rule (N/4) when we compute only over the positive magnetization sector...
    else:
        St_theta = np.sum(St_theta_Mat, axis=0)

    St_theta = St_theta.reshape((tgrid2.size, tgrid1.size))  # rows of St_theta indexed by t2 and columns indexed by t1 in line with Spinach convention

    ResponseFunc_Real_da = xr.DataArray(np.real(St_theta), coords=[tgrid2, tgrid1], dims=['t2', 't1'])
    ResponseFunc_Imag_da = xr.DataArray(np.imag(St_theta), coords=[tgrid2, tgrid1], dims=['t2', 't1'])

    data_dict = {'ResponseFunc_Real': ResponseFunc_Real_da, 'ResponseFunc_Imag': ResponseFunc_Imag_da}
    coords_dict = {'t1': tgrid1, 't2': tgrid2, 'basisStates_comp': basisStates_comp, 'basisStates_all': spinBasis.states, 'i': np.arange(spinBasis.L), 'j': np.arange(spinBasis.L)}
    attrs_dict = {'Nspins': N}

    trueED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trueED_ds


def trueSim_zerofield(tgrid, spinBasis, HParams, weights, shotNoiseParams, lowfield=False, posMag=True):
    ShotNoise = shotNoiseParams['ShotNoise']; N_ShotNoise = shotNoiseParams['N_ShotNoise']
    [JijList, hiList] = HParams
    H_theta = fullHamiltonian(spinBasis, JijList, hiList, chemicalShifts=lowfield)
    # H_theta = XYZHamiltonian(spinBasis, JijList, hiList, chemicalShifts=lowfield)
    diagH, P = H_theta.eigh()
    deltaE = np.around((diagH - diagH[0]) / (2 * np.pi * 136.2), 5); print(deltaE)
    # print(diagH / (2 * np.pi * 136.2))

    N = spinBasis.L

    pulseAngle = np.pi / 2

    ham_Rx = hamiltonian([["x", [[weights[i], i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rx, P_Rx = ham_Rx.eigh()
    ham_Ry = hamiltonian([["y", [[weights[i], i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.complex128, check_symm=False, check_herm=False); diagH_Ry, P_Ry = ham_Ry.eigh()
    ham_Rz = hamiltonian([["z", [[weights[i], i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rz, P_Rz = ham_Rz.eigh()

    U_Rx = P_Rx @ np.diag(np.exp(-1j * pulseAngle * diagH_Rx)) @ np.conj(P_Rx).T; U_Rx_m = P_Rx @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Rx)) @ np.conj(P_Rx).T
    U_Ry = P_Ry @ np.diag(np.exp(-1j * pulseAngle * diagH_Ry)) @ np.conj(P_Ry).T; U_Ry_m = P_Ry @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Ry)) @ np.conj(P_Ry).T
    U_Rz = P_Rz @ np.diag(np.exp(-1j * pulseAngle * diagH_Rz)) @ np.conj(P_Rz).T; U_Rz_m = P_Rz @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Rz)) @ np.conj(P_Rz).T

    # Sz_Tot = SzTot(spinBasis)
    Sz_Tot = np.diag(ham_Rz.todense())
    if posMag:
        magMask = np.logical_not(np.isclose(Sz_Tot, 0.0, atol=1e-3)) * (Sz_Tot > 0.0)
        Sz_Tot_comp = Sz_Tot[magMask]
        basisStates_comp = spinBasis.states[magMask]
    else:
        Sz_Tot_comp = Sz_Tot
        basisStates_comp = spinBasis.states

    St_theta_Mat = np.zeros((basisStates_comp.size, tgrid.size))
    entEntropy_Mat = np.zeros((basisStates_comp.size, tgrid.size))
    overlap_Mat = np.zeros((basisStates_comp.size, tgrid.size))

    for indb, bint in enumerate(basisStates_comp):
        bstate = basisVector(bint, spinBasis)

        psi_t = np.zeros((bstate.size, tgrid.size), dtype=complex)
        for indt, t in enumerate(tgrid):
            U_t = get_U(diagH, P, t)
            # psi_t[:, indt] = np.dot(U_t @ U_Ry, bstate)
            psi_indt = np.dot(U_t, bstate)
            psi_t[:, indt] = psi_indt
            # print(np.around(np.abs(psi_t[:, indt])**2, 3))
            entEntropy_Mat[indb, indt] = spinBasis.ent_entropy(state=psi_indt, sub_sys_A=[0, 1], density=True)['Sent_A']
            overlap_Mat[indb, indt] = np.abs(np.vdot(bstate, psi_indt))

        ave_Sz_Tot = aveSzTot(psi_t, Sz_Tot, ShotNoise, N_ShotNoise)
        # print(ave_Sz_Tot)
        St_theta_Mat[indb, :] = ave_Sz_Tot * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

    if posMag:
        St_theta = 2 * np.sum(St_theta_Mat, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector. For some reason St_theta[0] is a little off from the sum rule (N/4) when we compute only over the positive magnetization sector...
    else:
        St_theta = np.sum(St_theta_Mat, axis=0)

    ResponseFunc_da = xr.DataArray(St_theta, coords=[tgrid], dims=['t'])

    hamiltonian_da = xr.DataArray(hamiltonianListToMatrix(HParams, spinBasis.L, True), coords=[np.arange(spinBasis.L), np.arange(spinBasis.L)], dims=['i', 'j'])
    # entEntropy_da = xr.DataArray(entEntropy_Mat, coords=[basisStates_comp, tgrid], dims=['basisStates_comp', 't'])
    entEntropy_Vec = np.mean(entEntropy_Mat, axis=0)
    entEntropy_da = xr.DataArray(entEntropy_Vec, coords=[tgrid], dims=['t'])

    overlap_da = xr.DataArray(overlap_Mat, coords=[basisStates_comp, tgrid], dims=['basisStates_comp', 't'])
    overlap_Ave_da = xr.DataArray(np.mean(overlap_Mat, axis=0), coords=[tgrid], dims=['t'])

    data_dict = {'ResponseFunc': ResponseFunc_da, 'HamiltonianMatrix': hamiltonian_da, 'EntEntropy': entEntropy_da, 'overlap': overlap_da, 'overlap_Ave': overlap_Ave_da}
    coords_dict = {'t': tgrid, 'basisStates_comp': basisStates_comp, 'basisStates_all': spinBasis.states, 'i': np.arange(spinBasis.L), 'j': np.arange(spinBasis.L)}
    attrs_dict = {'Nspins': N}

    trueED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trueED_ds


def trueSim_zerofield_qT(tgrid, spinBasis, HParams, weights, gamma, decohType):
    [JijList, hiList] = HParams
    N = spinBasis.L

    H_theta = fullHamiltonian(spinBasis, JijList, hiList, chemicalShifts=False)
    diagH_theta, P_theta = H_theta.eigh()
    # diagH_theta, P_theta = np.linalg.eigh(H_theta.todense())

    H, c_op_list = makeGenerators(JijList, hiList, N, gamma, decohType)
    # H = qt.Qobj(H_theta_mat)
    # diagH, P = H.eigenstates(); P = [vec.full().flatten() for vec in P]
    diagH, P = np.linalg.eigh(H.full())

    # liouv_mat = qt.liouvillian(H, c_op_list)
    # eig_mat = liouv_mat.eigenenergies()
    # re_eigmat = np.real(eig_mat)
    # imag_eigmat = np.imag(eig_mat)
    # scale = (2 * np.pi * 136.2)
    # import matplotlib.pyplot as plt
    # fig = plt.figure(6)
    # if gamma != 0:
    #     plt.plot(re_eigmat / gamma, imag_eigmat / scale, 'k.')
    # else:
    #     plt.plot(re_eigmat, imag_eigmat / scale, 'k.')
    # plt.show()

    # print(diagH_theta - diagH_theta[0])
    # print(diagH - diagH[0])

    ham_Rz = hamiltonian([["z", [[weights[i], i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rz, P_Rz = ham_Rz.eigh()
    Sz_Tot = np.diag(ham_Rz.todense())

    magMask = np.logical_not(np.isclose(Sz_Tot, 0.0, atol=1e-3)) * (Sz_Tot > 0.0)
    Sz_Tot_comp = Sz_Tot[magMask]
    basisStates_comp = spinBasis.states[magMask]

    # basisStates_comp = [qt.tensor([qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0)]),
    #                     qt.tensor([qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 1)]),
    #                     qt.tensor([qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 1), qt.basis(2, 0)]),
    #                     qt.tensor([qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 1), qt.basis(2, 1)]),
    #                     qt.tensor([qt.basis(2, 0), qt.basis(2, 1), qt.basis(2, 0), qt.basis(2, 0)]),
    #                     qt.tensor([qt.basis(2, 0), qt.basis(2, 1), qt.basis(2, 0), qt.basis(2, 1)]),
    #                     qt.tensor([qt.basis(2, 1), qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0)]),
    #                     qt.tensor([qt.basis(2, 1), qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 1)])]
    # Sz_Tot_comp = np.array([1.6257476076555024, 1.3742523923444976, 0.6257476076555024, 0.3742523923444976, 0.6257476076555024, 0.3742523923444976, 0.6257476076555024, 0.3742523923444976])

    St_theta_Mat = np.zeros((len(basisStates_comp), tgrid.size))

    for indb, bint in enumerate(basisStates_comp):
        bstate = basisVector(bint, spinBasis)
        state_prob = np.zeros((bstate.size, tgrid.size), dtype=float)
        result = qt.mesolve(H, qt.Qobj(bstate) * qt.Qobj(bstate).dag(), tgrid, c_op_list)

        # state_prob = np.zeros((2**N, tgrid.size), dtype=float)
        # result = qt.mesolve(H, bint * bint.dag(), tgrid, c_op_list)
        for indt, t in enumerate(tgrid):
            state_prob[:, indt] = np.abs(np.diag(result.states[indt].full()))
            # U_t = get_U(diagH, P, t); state_prob[:, indt] = np.abs(np.dot(U_t, bstate))**2

        ave_Sz_Tot = np.sum(np.multiply(Sz_Tot[:, None], state_prob), axis=0)

        St_theta_Mat[indb, :] = ave_Sz_Tot * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

    St_theta = 2 * np.sum(St_theta_Mat, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector. For some reason St_theta[0] is a little off from the sum rule (N/4) when we compute only over the positive magnetization sector...

    ResponseFunc_da = xr.DataArray(St_theta, coords=[tgrid], dims=['t'])

    data_dict = {'ResponseFunc': ResponseFunc_da}
    coords_dict = {'t': tgrid}
    attrs_dict = {'Nspins': N}

    trueED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trueED_ds


def trotterSim_zerofield(tgrid, spinBasis, HParams, weights, shotNoiseParams, trotParamDict):
    ShotNoise = shotNoiseParams['ShotNoise']; N_ShotNoise = shotNoiseParams['N_ShotNoise']
    runAverage = trotParamDict['runAverage']
    [JijList, hiList] = HParams
    gateHamiltonianList, JijVals = gateHamiltonians(spinBasis, JijList, hiList, chemicalShifts=True)
    [HxxList, HyyList, HzzList, Hx] = gateHamiltonianList
    gateHamiltonianList_diag = diagHamList(gateHamiltonianList)

    N = spinBasis.L; trotParamDict['N'] = N
    # Sz_Tot = SzTot(spinBasis)
    ham_Rz = hamiltonian([["z", [[weights[i], i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rz, P_Rz = ham_Rz.eigh()
    Sz_Tot = np.diag(ham_Rz.todense())

    magMask = np.logical_not(np.isclose(Sz_Tot, 0.0, atol=1e-3)) * (Sz_Tot > 0.0)
    Sz_Tot_comp = Sz_Tot[magMask]
    basisStates_comp = spinBasis.states[magMask]

    gateBudget = trotParamDict['gateCount']; c2 = trotParamDict['c2']; intGateTime = trotParamDict['intGateTime']; singRotTime = trotParamDict['singRotTime']

    St_theta_Mat_trotter = np.zeros((basisStates_comp.size, tgrid.size))

    trotStep = int(gateBudget / (3 * len(JijVals)))
    theta_input_Array = np.zeros((tgrid.size, trotStep * 3 * len(HxxList)))

    if trotParamDict['heatingCorrection']:
        iAtimer = timer()
        for tind, t in enumerate(tgrid):
            theta_input_Array[tind, :] = inputAngle_precompute(t, gateHamiltonianList_diag, JijVals, trotParamDict)
        print('Input Angle Precomputation Time: {0}'.format(timer() - iAtimer))
    St_List = []
    for indr in np.arange(runAverage):
        # print('runInd: {0}'.format(indr))
        for indb, bint in enumerate(basisStates_comp):
            bstate = basisVector(bint, spinBasis)
            psi_t_trotter = []
            for tind, t in enumerate(tgrid):
                psi_temp, counter = trotter_equal(bstate, t, gateHamiltonianList_diag, JijVals, trotParamDict, theta_input_Array[tind, :])
                psi_t_trotter.append(np.array(psi_temp))
            psi_t_trotter = np.transpose(np.array(psi_t_trotter))
            ave_Sz_Tot_trotter = aveSzTot(psi_t_trotter, Sz_Tot, ShotNoise, N_ShotNoise)
            St_theta_Mat_trotter[indb, :] = ave_Sz_Tot_trotter * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

        gateCountExplicit = counter
        St_theta_trotter = 2 * np.sum(St_theta_Mat_trotter, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector.
        St_List.append(St_theta_trotter)

    St_trotter_mean = np.mean(np.array(St_List), axis=0)

    ResponseFunc_da = xr.DataArray(St_trotter_mean, coords=[tgrid], dims=['t'])

    data_dict = {'ResponseFunc': ResponseFunc_da}
    coords_dict = {'t': tgrid, 'basisStates_comp': basisStates_comp, 'basisStates_all': spinBasis.states, 'i': np.arange(spinBasis.L), 'j': np.arange(spinBasis.L)}
    attrs_dict = {'Nspins': N, 'c2': c2, 'gateCount': gateBudget, 'intGateTime': intGateTime, 'singRotTime': singRotTime}

    trotED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trotED_ds


def trueSim_zerofield_aveHam(tgrid, spinBasis, HParams, weights, shotNoiseParams, lowfield=False, posMag=True):
    ShotNoise = shotNoiseParams['ShotNoise']; N_ShotNoise = shotNoiseParams['N_ShotNoise']
    [JijList, hiList] = HParams
    H_theta = scHamiltonian(spinBasis, JijList, hiList, chemicalShifts=lowfield)
    H_orig = fullHamiltonian(spinBasis, JijList, hiList, chemicalShifts=lowfield)
    diagH, P = H_theta.eigh()
    diagH_orig, P_orig = H_orig.eigh()

    normH = np.linalg.norm(H_theta.todense())
    maxH = np.max(np.abs(diagH))

    N = spinBasis.L

    pulseAngle = np.pi / 2
    ham_Rx = hamiltonian([["x", [[1, i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rx, P_Rx = ham_Rx.eigh()
    ham_Ry = hamiltonian([["y", [[1, i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.complex128, check_symm=False, check_herm=False); diagH_Ry, P_Ry = ham_Ry.eigh()
    ham_Rz = hamiltonian([["z", [[1, i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rz, P_Rz = ham_Rz.eigh()
    Rx = P_Rx @ np.diag(np.exp(-1j * pulseAngle * diagH_Rx)) @ np.conj(P_Rx).T; Rx_m = P_Rx @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Rx)) @ np.conj(P_Rx).T
    Ry = P_Ry @ np.diag(np.exp(-1j * pulseAngle * diagH_Ry)) @ np.conj(P_Ry).T; Ry_m = P_Ry @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Ry)) @ np.conj(P_Ry).T
    Rz = P_Rz @ np.diag(np.exp(-1j * pulseAngle * diagH_Rz)) @ np.conj(P_Rz).T; Rz_m = P_Rz @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Rz)) @ np.conj(P_Rz).T

    ham_Rz_weighted = hamiltonian([["z", [[weights[i], i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False)

    # Sz_Tot = SzTot(spinBasis)
    Sz_Tot = np.diag(ham_Rz_weighted.todense())
    if posMag:
        magMask = np.logical_not(np.isclose(Sz_Tot, 0.0, atol=1e-3)) * (Sz_Tot > 0.0)
        Sz_Tot_comp = Sz_Tot[magMask]
        basisStates_comp = spinBasis.states[magMask]
    else:
        Sz_Tot_comp = Sz_Tot
        basisStates_comp = spinBasis.states

    St_theta_Mat = np.zeros((basisStates_comp.size, tgrid.size))

    cycleRed = 100; timestep = tgrid[1] - tgrid[0]; T = timestep / cycleRed; tau = T / 6
    U_tau = get_U(diagH, P, tau)
    U_T = Ry @ Ry @ U_tau @ Rx @ U_tau @ Ry_m @ U_tau @ U_tau @ Ry_m @ U_tau @ Rx @ U_tau  # time-evolution for a total cycle time T = dt / cycleRed. Symmetric Heisenberg + x-disorder
    U_dt = np.linalg.matrix_power(U_T, cycleRed)  # time-evolution for a step dt in the time grid

    for indb, bint in enumerate(basisStates_comp):
        bstate = basisVector(bint, spinBasis)

        U_t = np.eye(U_T.shape[0])
        psi_t = np.zeros((bstate.size, tgrid.size), dtype=complex)
        for indt, t in enumerate(tgrid):

            # U_tau_orig = get_U(diagH_orig, P_orig, t / 6)
            # psi_t[:, indt] = np.dot(U_tau_orig @ U_tau_orig @ U_tau_orig @ U_tau_orig @ U_tau_orig @ U_tau_orig, bstate)

            # U_t = np.linalg.matrix_power(U_dt, indt)
            psi_t[:, indt] = np.dot(U_t, bstate)
            U_t = U_dt @ U_t

        ave_Sz_Tot = aveSzTot(psi_t, Sz_Tot, ShotNoise, N_ShotNoise)
        # print(ave_Sz_Tot)
        St_theta_Mat[indb, :] = ave_Sz_Tot * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

    if posMag:
        St_theta = 2 * np.sum(St_theta_Mat, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector. For some reason St_theta[0] is a little off from the sum rule (N/4) when we compute only over the positive magnetization sector...
    else:
        St_theta = np.sum(St_theta_Mat, axis=0)

    ResponseFunc_da = xr.DataArray(St_theta, coords=[tgrid], dims=['t'])

    hamiltonian_da = xr.DataArray(hamiltonianListToMatrix(HParams, spinBasis.L, True), coords=[np.arange(spinBasis.L), np.arange(spinBasis.L)], dims=['i', 'j'])

    data_dict = {'ResponseFunc': ResponseFunc_da, 'HamiltonianMatrix': hamiltonian_da}
    coords_dict = {'t': tgrid, 'basisStates_comp': basisStates_comp, 'basisStates_all': spinBasis.states, 'i': np.arange(spinBasis.L), 'j': np.arange(spinBasis.L)}
    attrs_dict = {'Nspins': N}

    trueED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trueED_ds


def trueSim_zerofield_fromFile(tgrid, spinBasis, HParams, weights, shotNoiseParams, lowfield=False, posMag=True):
    import syn_funcs as sf
    import cirq
    ShotNoise = shotNoiseParams['ShotNoise']; N_ShotNoise = shotNoiseParams['N_ShotNoise']
    [JijList, hiList] = HParams
    H_theta = fullHamiltonian(spinBasis, JijList, hiList, chemicalShifts=lowfield)
    diagH, P = H_theta.eigh()

    N = spinBasis.L

    pulseAngle = np.pi / 2

    ham_Rx = hamiltonian([["x", [[weights[i], i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rx, P_Rx = ham_Rx.eigh()
    ham_Ry = hamiltonian([["y", [[weights[i], i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.complex128, check_symm=False, check_herm=False); diagH_Ry, P_Ry = ham_Ry.eigh()
    ham_Rz = hamiltonian([["z", [[weights[i], i] for i in np.arange(N)]]], [], basis=spinBasis, dtype=np.float64, check_symm=False, check_herm=False); diagH_Rz, P_Rz = ham_Rz.eigh()

    U_Rx = P_Rx @ np.diag(np.exp(-1j * pulseAngle * diagH_Rx)) @ np.conj(P_Rx).T; U_Rx_m = P_Rx @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Rx)) @ np.conj(P_Rx).T
    U_Ry = P_Ry @ np.diag(np.exp(-1j * pulseAngle * diagH_Ry)) @ np.conj(P_Ry).T; U_Ry_m = P_Ry @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Ry)) @ np.conj(P_Ry).T
    U_Rz = P_Rz @ np.diag(np.exp(-1j * pulseAngle * diagH_Rz)) @ np.conj(P_Rz).T; U_Rz_m = P_Rz @ np.diag(np.exp(-1j * (-1 * pulseAngle) * diagH_Rz)) @ np.conj(P_Rz).T

    print(spinBasis.states)
    print(SzTot(spinBasis))
    # Sz_Tot = SzTot(spinBasis)
    Sz_Tot = np.diag(ham_Rz.todense())
    if posMag:
        magMask = np.logical_not(np.isclose(Sz_Tot, 0.0, atol=1e-3)) * (Sz_Tot > 0.0)
        Sz_Tot_comp = Sz_Tot[magMask]
        basisStates_comp = spinBasis.states[magMask]
    else:
        Sz_Tot_comp = Sz_Tot
        basisStates_comp = spinBasis.states

    St_theta_Mat = np.zeros((basisStates_comp.size, tgrid.size))

    for indb, bint in enumerate(basisStates_comp):
        bstate = basisVector(bint, spinBasis)

        psi_t = np.zeros((bstate.size, tgrid.size), dtype=complex)
        for indt, t in enumerate(tgrid):
            if indt == 0:
                U_t = np.eye(bstate.size)
            else:
                fileout = '/Users/kis/Dropbox/VariationalResearch/HarvardOdyssey/NMR_data/Output/circuit_synthesis/acetonitrile/zf/cirq_threshold_5e-2/U_t_{:d}.pickle'.format(indt)
                cirquit_import = sf.load_circuit(fileout)
                U_t = cirq.unitary(cirquit_import)
                # print(indt, U_t.shape, len(cirquit_import.all_qubits()))
            # U_t = get_U(diagH, P, t)
            psi_t[:, indt] = np.dot(U_t, bstate)

        ave_Sz_Tot = aveSzTot(psi_t, Sz_Tot, ShotNoise, N_ShotNoise)
        print(bint)
        print(bstate)
        print(np.real(np.abs(psi_t)**2)[:, 0])
        print(ave_Sz_Tot[0])
        St_theta_Mat[indb, :] = ave_Sz_Tot * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

    if posMag:
        St_theta = 2 * np.sum(St_theta_Mat, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector. For some reason St_theta[0] is a little off from the sum rule (N/4) when we compute only over the positive magnetization sector...
    else:
        St_theta = np.sum(St_theta_Mat, axis=0)

    ResponseFunc_da = xr.DataArray(St_theta, coords=[tgrid], dims=['t'])

    hamiltonian_da = xr.DataArray(hamiltonianListToMatrix(HParams, spinBasis.L, True), coords=[np.arange(spinBasis.L), np.arange(spinBasis.L)], dims=['i', 'j'])

    data_dict = {'ResponseFunc': ResponseFunc_da, 'HamiltonianMatrix': hamiltonian_da}
    coords_dict = {'t': tgrid, 'basisStates_comp': basisStates_comp, 'basisStates_all': spinBasis.states, 'i': np.arange(spinBasis.L), 'j': np.arange(spinBasis.L)}
    attrs_dict = {'Nspins': N}

    trueED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trueED_ds


def trueSim_fromFile(cluster, t_ind_List, tgrid_tot, spinBasis, HParams, shotNoiseParams, decayRate, posMag=True, saveAllStates=False):
    import syn_funcs as sf
    import cirq
    ShotNoise = shotNoiseParams['ShotNoise']; N_ShotNoise = shotNoiseParams['N_ShotNoise']
    tgrid = tgrid_tot[t_ind_List]
    N = spinBasis.L

    pulseAngle = np.pi / 2

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

    if saveAllStates:
        psiFinal_Real_da = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns, tgrid.size), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states, tgrid], dims=['basisStates_comp', 'basisStates_all', 't'])
        psiFinal_Imag_da = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns, tgrid.size), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states, tgrid], dims=['basisStates_comp', 'basisStates_all', 't'])
    else:
        psiFinal_Real_da = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states], dims=['basisStates_comp', 'basisStates_all'])
        psiFinal_Imag_da = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states], dims=['basisStates_comp', 'basisStates_all'])

    for indb, bint in enumerate(basisStates_comp):
        bstate = basisVector(bint, spinBasis)

        psi_t_re = np.zeros((bstate.size, tgrid.size), dtype=complex)
        psi_t_im = np.zeros((bstate.size, tgrid.size), dtype=complex)
        for indt, t in enumerate(tgrid):
            fileout = "syn_qasm/syn_cirq/U" + cluster + "_t_{:d}.pickle".format(t_ind_List[indt])
            cirquit_import = sf.load_circuit(fileout)
            U_theta = cirq.unitary(cirquit_import)
            psi_t_re[:, indt] = np.dot(U_theta, bstate)
            psi_t_im[:, indt] = np.dot(U_Rx @ U_theta, bstate)

            # psi_t_re[:, indt] = np.dot(U_Rz_m @ U_Rx_m @ U_theta, bstate)
            # psi_t_im[:, indt] = np.dot(U_Ry @ U_theta, bstate)

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


def trotterSim_fixedGC(tgrid, spinBasis, HParams, trotParamDict, shotNoiseParams, decayRate, posMag=True, saveAllStates=False):
    # posMag=True does the computation for only positive magnetization basis states: this is sufficient to calculate the response function and spectrum, but not to calculate the average fidelity.
    # saveAllStates=True saves the full time-evolved system state for all time samples of S(t). If it is set to False, then it only saves the state at the last time
    ShotNoise = shotNoiseParams['ShotNoise']; N_ShotNoise = shotNoiseParams['N_ShotNoise']
    runAverage = trotParamDict['runAverage']
    [JijList, hiList] = HParams
    gateHamiltonianList, JijVals = gateHamiltonians(spinBasis, JijList, hiList, chemicalShifts=True)
    [HxxList, HyyList, HzzList, Hx] = gateHamiltonianList
    gateHamiltonianList_diag = diagHamList(gateHamiltonianList)
    # for ind, Hxx in enumerate(HxxList):
    #     print(HyyList[ind])
    #     print(JijVals[ind] / 4)

    N = spinBasis.L; trotParamDict['N'] = N
    Sz_Tot = SzTot(spinBasis)
    if posMag:
        magMask = np.logical_not(np.isclose(Sz_Tot, 0.0, atol=1e-3)) * (Sz_Tot > 0.0)
        Sz_Tot_comp = Sz_Tot[magMask]
        basisStates_comp = spinBasis.states[magMask]
    else:
        Sz_Tot_comp = Sz_Tot
        basisStates_comp = spinBasis.states

    gateBudget = trotParamDict['gateCount']; c2 = trotParamDict['c2']; intGateTime = trotParamDict['intGateTime']; singRotTime = trotParamDict['singRotTime']

    St_theta_Mat_trotter = np.zeros((basisStates_comp.size, tgrid.size))

    trueED_ds = trueSim(tgrid, spinBasis, HParams, {'ShotNoise': False, 'N_ShotNoise': 10000}, decayRate, posMag, saveAllStates)

    if saveAllStates:
        psiFinal_Real_da = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns, tgrid.size), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states, tgrid], dims=['basisStates_comp', 'basisStates_all', 't'])
        psiFinal_Imag_da = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns, tgrid.size), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states, tgrid], dims=['basisStates_comp', 'basisStates_all', 't'])
    else:
        psiFinal_Real_da = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states], dims=['basisStates_comp', 'basisStates_all'])
        psiFinal_Imag_da = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states], dims=['basisStates_comp', 'basisStates_all'])

    trotStep = int(gateBudget / (3 * len(JijVals) + 2 * N))
    theta_input_Array = np.zeros((tgrid.size, trotStep * 3 * len(HxxList)))
    if trotParamDict['heatingCorrection']:
        iAtimer = timer()
        for tind, t in enumerate(tgrid):
            theta_input_Array[tind, :] = inputAngle_precompute(t, gateHamiltonianList_diag, JijVals, trotParamDict)
        print('Input Angle Precomputation Time: {0}'.format(timer() - iAtimer))
    St_List = []
    hellDist_List = []
    aveFid_List = []
    for indr in np.arange(runAverage):
        # print('runInd: {0}'.format(indr))
        if saveAllStates:
            psiFinal_Real_da_temp = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns, tgrid.size), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states, tgrid], dims=['basisStates_comp', 'basisStates_all', 't'])
            psiFinal_Imag_da_temp = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns, tgrid.size), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states, tgrid], dims=['basisStates_comp', 'basisStates_all', 't'])
        else:
            psiFinal_Real_da_temp = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states], dims=['basisStates_comp', 'basisStates_all'])
            psiFinal_Imag_da_temp = xr.DataArray(np.full((basisStates_comp.size, spinBasis.Ns), np.nan, dtype=float), coords=[basisStates_comp, spinBasis.states], dims=['basisStates_comp', 'basisStates_all'])

        for indb, bint in enumerate(basisStates_comp):
            bstate = basisVector(bint, spinBasis)
            # print(bint, Sz_Tot_comp[indb], bstate)  # integer representation of basis state, total magnetization associated with the state, and vector representation of the state
            psi_t_trotter = []
            for tind, t in enumerate(tgrid):
                psi_temp, counter = trotter_equal(bstate, t, gateHamiltonianList_diag, JijVals, trotParamDict, theta_input_Array[tind, :])
                # if trotParamDict['trotOrder'] == 'Old':
                #     psi_temp, counter = trotter1_fixedGC_precompute_old(bstate, t, gateHamiltonianList_diag, JijVals, trotParamDict)
                # else:
                #     psi_temp, counter = trotter1_fixedGC_precompute(bstate, t, gateHamiltonianList_diag, JijVals, trotParamDict)
                psi_t_trotter.append(np.array(psi_temp))
            psi_t_trotter = np.transpose(np.array(psi_t_trotter))
            ave_Sz_Tot_trotter = aveSzTot(psi_t_trotter, Sz_Tot, ShotNoise, N_ShotNoise)
            St_theta_Mat_trotter[indb, :] = ave_Sz_Tot_trotter * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

            if saveAllStates:
                psiFinal_Real_da_temp[indb] = np.real(psi_t_trotter)
                psiFinal_Imag_da_temp[indb] = np.imag(psi_t_trotter)
            else:
                psi_t_trotter_final = psi_t_trotter[:, -1]
                psiFinal_Real_da_temp[indb] = np.real(psi_t_trotter_final)
                psiFinal_Imag_da_temp[indb] = np.imag(psi_t_trotter_final)

        gateCountExplicit = counter
        # print(gateBudget, gateCountExplicit)
        # St_theta_trotter = 2 * np.sum(St_theta_Mat_trotter, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector.
        if posMag:
            St_theta_trotter = 2 * np.sum(St_theta_Mat_trotter, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector.
        else:
            St_theta_trotter = np.sum(St_theta_Mat_trotter, axis=0)
        St_List.append(St_theta_trotter)

        Aw_trotter_temp, fVals_trotter = spectFunc(St_theta_trotter, tgrid, decayRate)
        Spectrum_da = xr.DataArray(Aw_trotter_temp, coords=[fVals_trotter], dims=['f'])
        trot_ds_temp = xr.Dataset({'Spectrum': Spectrum_da, 'psiFinal_Real': psiFinal_Real_da_temp, 'psiFinal_Imag': psiFinal_Imag_da_temp}, coords={'f': fVals_trotter, 't': tgrid, 'basisStates_comp': basisStates_comp, 'basisStates_all': spinBasis.states}, attrs={})
        hellDist, aveFid = simComp(trueED_ds, trot_ds_temp, saveAllStates)
        hellDist_List.append(hellDist)
        aveFid_List.append(aveFid)

    St_trotter_mean = np.mean(np.array(St_List), axis=0)
    hellDist_mean = np.mean(np.array(hellDist_List))  # This is a disorder average of Hellinger Distances rather than the Hellinger Distance of a disorder averaged spectrum. The two limits don't commute so we shouldn't use this.
    aveFid_mean = np.mean(np.array(aveFid_List), axis=0)

    psiFinal_Real_da = psiFinal_Real_da_temp; psiFinal_Imag_da = psiFinal_Imag_da_temp  # Just saves the last computed value

    Aw_theta_trotter, fVals_trotter = spectFunc(St_trotter_mean, tgrid, decayRate)
    # print(N / 4, St_theta_trotter[0], 0.5 * simps(Aw_theta_trotter, dx=(fVals_trotter[1] - fVals_trotter[0])))  # checks sum rule (spectra Aw_theta may not satisfy it due to resolution in frequency and decay we input by hand)

    Spectrum_da = xr.DataArray(Aw_theta_trotter, coords=[fVals_trotter], dims=['f'])
    ResponseFunc_da = xr.DataArray(St_trotter_mean, coords=[tgrid], dims=['t'])
    AveFid_da = xr.DataArray(aveFid_mean, coords=[tgrid], dims=['t'])
    hamiltonian_da = xr.DataArray(hamiltonianListToMatrix(HParams, spinBasis.L, True), coords=[np.arange(spinBasis.L), np.arange(spinBasis.L)], dims=['i', 'j'])

    data_dict = {'Spectrum': Spectrum_da, 'ResponseFunc': ResponseFunc_da, 'AveFid': AveFid_da, 'psiFinal_Real': psiFinal_Real_da, 'psiFinal_Imag': psiFinal_Imag_da, 'HamiltonianMatrix': hamiltonian_da}
    coords_dict = {'f': fVals_trotter, 't': tgrid, 'basisStates_comp': basisStates_comp, 'basisStates_all': spinBasis.states, 'i': np.arange(spinBasis.L), 'j': np.arange(spinBasis.L)}
    attrs_dict = {'Nspins': N, 'c2': c2, 'gateCount': gateBudget, 'intGateTime': intGateTime, 'singRotTime': singRotTime}

    trotED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trotED_ds


def trotterSim_fixedGC_specOnly(tgrid, spinBasis, HParams, trotParamDict, shotNoiseParams, decayRate):
    ShotNoise = shotNoiseParams['ShotNoise']; N_ShotNoise = shotNoiseParams['N_ShotNoise']
    runAverage = trotParamDict['runAverage']
    [JijList, hiList] = HParams
    gateHamiltonianList, JijVals = gateHamiltonians(spinBasis, JijList, hiList, chemicalShifts=True)
    [HxxList, HyyList, HzzList, Hx] = gateHamiltonianList
    gateHamiltonianList_diag = diagHamList(gateHamiltonianList)

    N = spinBasis.L; trotParamDict['N'] = N
    Sz_Tot = SzTot(spinBasis)

    magMask = np.logical_not(np.isclose(Sz_Tot, 0.0, atol=1e-3)) * (Sz_Tot > 0.0)
    Sz_Tot_comp = Sz_Tot[magMask]
    basisStates_comp = spinBasis.states[magMask]

    gateBudget = trotParamDict['gateCount']; c2 = trotParamDict['c2']; intGateTime = trotParamDict['intGateTime']; singRotTime = trotParamDict['singRotTime']

    St_theta_Mat_trotter = np.zeros((basisStates_comp.size, tgrid.size))

    trotStep = int(gateBudget / (3 * len(JijVals) + 2 * N))
    theta_input_Array = np.zeros((tgrid.size, trotStep * 3 * len(HxxList)))
    if trotParamDict['heatingCorrection']:
        iAtimer = timer()
        for tind, t in enumerate(tgrid):
            theta_input_Array[tind, :] = inputAngle_precompute(t, gateHamiltonianList_diag, JijVals, trotParamDict)
        print('Input Angle Precomputation Time: {0}'.format(timer() - iAtimer))
    St_List = []
    for indr in np.arange(runAverage):
        # print('runInd: {0}'.format(indr))
        for indb, bint in enumerate(basisStates_comp):
            bstate = basisVector(bint, spinBasis)
            psi_t_trotter = []
            for tind, t in enumerate(tgrid):
                psi_temp, counter = trotter_equal(bstate, t, gateHamiltonianList_diag, JijVals, trotParamDict, theta_input_Array[tind, :])
                psi_t_trotter.append(np.array(psi_temp))
            psi_t_trotter = np.transpose(np.array(psi_t_trotter))
            ave_Sz_Tot_trotter = aveSzTot(psi_t_trotter, Sz_Tot, ShotNoise, N_ShotNoise)
            St_theta_Mat_trotter[indb, :] = ave_Sz_Tot_trotter * Sz_Tot_comp[indb] / spinBasis.Ns  # each row is a term in the sum (=from one basis state) of S(t|\theta)

        gateCountExplicit = counter
        St_theta_trotter = 2 * np.sum(St_theta_Mat_trotter, axis=0)  # sums all terms (each term is the contribution from an initial basis state). Factor of 2 comes from the negative magnetization sector.
        St_List.append(St_theta_trotter)

    St_trotter_mean = np.mean(np.array(St_List), axis=0)

    Aw_theta_trotter, fVals_trotter = spectFunc(St_trotter_mean, tgrid, decayRate)

    Spectrum_da = xr.DataArray(Aw_theta_trotter, coords=[fVals_trotter], dims=['f'])
    ResponseFunc_da = xr.DataArray(St_trotter_mean, coords=[tgrid], dims=['t'])
    hamiltonian_da = xr.DataArray(hamiltonianListToMatrix(HParams, spinBasis.L, True), coords=[np.arange(spinBasis.L), np.arange(spinBasis.L)], dims=['i', 'j'])

    data_dict = {'Spectrum': Spectrum_da, 'ResponseFunc': ResponseFunc_da, 'HamiltonianMatrix': hamiltonian_da}
    coords_dict = {'f': fVals_trotter, 't': tgrid, 'basisStates_comp': basisStates_comp, 'basisStates_all': spinBasis.states, 'i': np.arange(spinBasis.L), 'j': np.arange(spinBasis.L)}
    attrs_dict = {'Nspins': N, 'c2': c2, 'gateCount': gateBudget, 'intGateTime': intGateTime, 'singRotTime': singRotTime}

    trotED_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return trotED_ds


def paramMatToHParams(paramMat):
    N = paramMat.shape[0]
    hiList = [[paramMat[i, i], i] for i in np.arange(N)]  # extracts hi from parameter matrix (puts in form for QuSpin)
    JijList = [[2 * paramMat[i, j], i, j] for i in np.arange(N) for j in np.arange(N) if (i != j) and (i < j) if not np.isclose(paramMat[i, j], 0)]  # extracts Jij from parameter matrix (puts in form for QuSpin); this list combines the Jij and Jji terms (Hermitian conjugates) into a single term
    spinBasis = spin_basis_1d(N, pauli=False)
    HParams = [JijList, hiList]
    return HParams, spinBasis


def sampleToHamiltonianParameters(sampleVec, rvIndices, factor_of_2):
    # takes a vector of parameters representing a single sample of parameter space (corresponding to one Hamiltonian) and outputs the corresponding Hamiltonian
    # Note: the factor_of_2 decides whether we want to simulate unitary evolution more cheaply by realizing Jij = Jji so we combine Jij*Si*Sj + Jji*Sj*Si = 2*Jij*Si*Sj
    if factor_of_2:
        f2 = 2
    else:
        f2 = 1

    hiList = []
    JijList = []
    for ind, rtup in enumerate(rvIndices):
        if np.isclose(sampleVec[ind], 0):
            continue
        [i, j] = rtup
        if i == j:
            hiList.append([sampleVec[ind], i])
        elif i < j:
            JijList.append([f2 * sampleVec[ind], i, j])
        else:
            print('INPUT ERROR')
    HParams = [JijList, hiList]
    return HParams


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


def bayesianWeights(trueSpectrum_norm, fVals, spectrumSamples):
    spectrumSamples_norm = []
    for ind, spectrum in enumerate(spectrumSamples):
        spectrumSamples_norm.append(spectrum / simps(y=spectrum, x=fVals))

    denom = np.sum(spectrumSamples_norm, axis=0) / len(spectrumSamples_norm)
    zeroMask = np.isclose(denom, 0)
    weights = []
    for indn, spectrum_norm in enumerate(spectrumSamples_norm):
        integrand = trueSpectrum_norm * spectrum_norm / denom
        integrand[zeroMask] = 0
        weights.append(simps(y=integrand, x=fVals))
    return np.array(weights)


def covarianceRegularization(sampleCov, covRegParam):
    # takes a covaraince matrix (generated from samples) and regularizes it so that its large eigenvalues are thresholded from being too large
    # sampleCov += (1e-5) * np.min(np.abs(sampleCov)) * np.eye(sampleCov.shape[0])
    # sampleCov_reg_1 = np.linalg.inv(np.linalg.inv(sampleCov) + covRegParam * np.eye(sampleCov.shape[0]))
    sampleCov_reg = sampleCov - sampleCov @ np.linalg.inv(sampleCov + np.eye(sampleCov.shape[0]) / covRegParam) @ sampleCov
    # print((sampleCov_reg - sampleCov_reg_1) / sampleCov_reg)
    return sampleCov_reg


def bayesianInference(neighborMats, trotParamDict, refSpectrum_ds, gridParams, inferenceParams, shotNoiseParams):
    Aw_true = refSpectrum_ds['Spectrum'].values; fVals_true = refSpectrum_ds.coords['f'].values
    Aw_true_norm = Aw_true / simps(y=Aw_true, x=fVals_true)
    spinBasis = gridParams['spinBasis']; tgrid = gridParams['tgrid']
    NSamples = inferenceParams['NSamples']; NIterations = inferenceParams['NIterations']; simType = inferenceParams['simType']; sampleMemory = inferenceParams['sampleMemory']; sampleDiscount = inferenceParams['sampleDiscount']
    covRegParam = inferenceParams['covRegParam']; averagePrior = inferenceParams['averagePrior']; decayRate_true = inferenceParams['decayRate_true']; decayRate_trot = inferenceParams['decayRate_trot']; decayRate_trot_noHeating = inferenceParams['decayRate_trot_noHeating']
    f2_bool = True

    gateBudget = trotParamDict['gateCount']; c2 = trotParamDict['c2']; intGateTime = trotParamDict['intGateTime']; singRotTime = trotParamDict['singRotTime']

    hellDist_mean_da = xr.DataArray(np.zeros(NIterations + 1), coords=[np.arange(NIterations + 1)], dims=['iteration'])
    Spectrum_mean_da = xr.DataArray(np.zeros((NIterations + 1, fVals_true.size)), coords=[np.arange(NIterations + 1), fVals_true], dims=['iteration', 'f'])
    hellDist_da = xr.DataArray(np.zeros(NIterations + 1), coords=[np.arange(NIterations + 1)], dims=['iteration'])
    hellDist_MeanOfSamples_da = xr.DataArray(np.zeros(NIterations + 1), coords=[np.arange(NIterations + 1)], dims=['iteration'])
    hellDist_VarOfSamples_da = xr.DataArray(np.zeros(NIterations + 1), coords=[np.arange(NIterations + 1)], dims=['iteration'])
    Spectrum_da = xr.DataArray(np.zeros((NIterations + 1, fVals_true.size)), coords=[np.arange(NIterations + 1), fVals_true], dims=['iteration', 'f'])
    hamiltonian_da = xr.DataArray(np.full((NIterations + 1, spinBasis.L, spinBasis.L), np.nan, dtype=float), coords=[np.arange(NIterations + 1), np.arange(spinBasis.L), np.arange(spinBasis.L)], dims=['iteration', 'i', 'j'])
    hamiltonian_mean_da = xr.DataArray(np.full((NIterations + 1, spinBasis.L, spinBasis.L), np.nan, dtype=float), coords=[np.arange(NIterations + 1), np.arange(spinBasis.L), np.arange(spinBasis.L)], dims=['iteration', 'i', 'j'])
    hamDA = []

    # Compute initial atomic prior
    N_Neighbors = len(neighborMats)
    atomicPriorMolecules = []
    rvIndices = []
    for mind in np.arange(N_Neighbors):
        nmat = neighborMats[mind]
        matParamVec = []
        for i in np.arange(spinBasis.N):
            for j in np.arange(spinBasis.N):
                if (i < j) or (i == j):
                    matParamVec.append(nmat[i, j])
        atomicPriorMolecules.append(matParamVec)

    for i in np.arange(spinBasis.N):
        for j in np.arange(spinBasis.N):
            if (i < j) or (i == j):
                rvIndices.append([i, j])

    atomicPriorIndices = np.random.randint(low=0, high=N_Neighbors, size=NSamples)
    atomicPriorSamples = [atomicPriorMolecules[i] for i in atomicPriorIndices]
    # print(atomicPriorSamples)

    # Compute initial posterior update
    spectrumSamples = []
    hDSamples = []
    hamSamples = []

    if averagePrior:
        for indm, nmol in enumerate(atomicPriorMolecules):
            sampleVec = nmol
            HParams0 = sampleToHamiltonianParameters(sampleVec, rvIndices, f2_bool)
            # simstart = timer()
            if simType == 'True':
                spectED_ds = trueSim(tgrid, spinBasis, HParams0, shotNoiseParams, decayRate_true)
            elif simType == 'Trotter':
                spectED_ds = trotterSim_fixedGC_specOnly(tgrid, spinBasis, HParams0, trotParamDict, shotNoiseParams, decayRate_trot)
            else:
                print('SIM TYPE ERROR')
            # simend = timer()
            # print(simend - simstart)
            spectrumSamples.append(spectED_ds['Spectrum'].values)
            hDSamples.append(hellingerDistance(spectED_ds['Spectrum'].values, Aw_true, fVals_true))
            hamSamples.append(HParams0)
        atomicPriorSamplesMat = np.stack(atomicPriorMolecules)
    else:
        for ns in np.arange(NSamples):
            sampleVec = atomicPriorSamples[ns]
            HParams0 = sampleToHamiltonianParameters(sampleVec, rvIndices, f2_bool)
            # simstart = timer()
            if simType == 'True':
                spectED_ds = trueSim(tgrid, spinBasis, HParams0, shotNoiseParams, decayRate_true)
            elif simType == 'Trotter':
                spectED_ds = trotterSim_fixedGC_specOnly(tgrid, spinBasis, HParams0, trotParamDict, shotNoiseParams, decayRate_trot)
            else:
                print('SIM TYPE ERROR')
            # simend = timer()
            # print(simend - simstart)
            spectrumSamples.append(spectED_ds['Spectrum'].values)
            hDSamples.append(hellingerDistance(spectED_ds['Spectrum'].values, Aw_true, fVals_true))
            hamSamples.append(HParams0)

        atomicPriorSamplesMat = np.stack(atomicPriorSamples)

    hDmin_ind = np.argmin(np.array(hDSamples))
    Spectrum_da[0, :] = spectrumSamples[hDmin_ind]
    hellDist_da[0] = hDSamples[hDmin_ind]
    hamiltonian_da[0] = hamiltonianListToMatrix(hamSamples[hDmin_ind], spinBasis.L, f2_bool)
    hellDist_MeanOfSamples_da[0] = np.mean(np.array(hDSamples))
    hellDist_VarOfSamples_da[0] = np.var(np.array(hDSamples))

    atomicSampleMean = np.mean(atomicPriorSamplesMat, axis=0)
    print(atomicSampleMean); print('\n')

    HParams_mean = sampleToHamiltonianParameters(atomicSampleMean, rvIndices, f2_bool)
    if simType == 'True':
        spectED_mean_ds = trueSim(tgrid, spinBasis, HParams_mean, shotNoiseParams, decayRate_true)
    elif simType == 'Trotter':
        spectED_mean_ds = trotterSim_fixedGC_specOnly(tgrid, spinBasis, HParams_mean, trotParamDict, shotNoiseParams, decayRate_trot)
    else:
        print('SIM TYPE ERROR')
    hellDist_mean_da[0] = hellingerDistance(spectED_mean_ds['Spectrum'].values, Aw_true, fVals_true)
    Spectrum_mean_da[0, :] = spectED_mean_ds['Spectrum'].values
    hamiltonian_mean_da[0] = hamiltonianListToMatrix(HParams_mean, spinBasis.L, f2_bool)

    print('Initial Hellinger Distance: {0}'.format(hellDist_mean_da[0].values))

    updateWeights = bayesianWeights(Aw_true_norm, fVals_true, spectrumSamples)
    # posteriorSamplesMat = np.transpose(atomicPriorSamplesMat * updateWeights[:, np.newaxis])  # each row represents sample values of one variable (e.g. Jij or hii)
    # sampleMean = np.mean(posteriorSamplesMat, axis=1)  # this is equivalent to \sum[updateWeights * priorSamples]/ \sum[updateWeights]
    # sampleCov = np.cov(posteriorSamplesMat)  # this is not quite the same as the weighted update below

    sampleMean = np.sum(np.transpose(atomicPriorSamplesMat * updateWeights[:, np.newaxis]), axis=1) / np.sum(updateWeights)
    sampleCov = np.cov(np.transpose(atomicPriorSamplesMat), aweights=updateWeights)  # this is equivalent to \sum[updateWeights * (priorSamples - weightedSampleMean)^{T} * (priorSamples - weightedSampleMean)] / (\sum[updateWeights] - \sum[updateWeights^2]/sum[updateWeights])
    covNorm = np.sum(updateWeights) - np.sum(updateWeights**2) / np.sum(updateWeights); covFix = covNorm / np.sum(updateWeights)
    sampleCov = covFix * sampleCov
    sampleCov = covarianceRegularization(sampleCov, covRegParam)

    # meanZeroPrior = atomicPriorSamplesMat - sampleMean
    # sampleCov_explicit = np.dot(np.transpose(meanZeroPrior * updateWeights[:, np.newaxis]), meanZeroPrior) / np.sum(updateWeights)  # explicit computation of the sample covariance using the prior samples and update weights instead of the np.cov function
    # print(np.all(np.isclose(sampleCov - sampleCov_explicit, 0)))

    print(sampleMean)
    print(np.sqrt(np.diag(sampleCov)))
    print(np.abs(np.sqrt(np.diag(sampleCov)) / sampleMean))
    # print(np.sum(updateWeights**2) / (np.sum(updateWeights)**2))
    print('\n')

    posteriorRV = multivariate_normal(mean=sampleMean, cov=sampleCov, allow_singular=True)
    posteriorSamples = posteriorRV.rvs(size=NSamples)

    sampleMemory_temp = copy(sampleMemory)
    if sampleMemory > 0:
        runningSamples = []
        if averagePrior:
            for inda, ap in enumerate(atomicPriorMolecules):
                runningSamples.append(np.array(ap))
        else:
            for inda, ap in enumerate(atomicPriorSamples):
                runningSamples.append(np.array(ap))
        runningWeights = [*updateWeights]

    for iind in np.arange(NIterations):
        priorSamples = posteriorSamples
        spectrumSamples = []
        hDSamples = []
        hamSamples = []
        for ns in np.arange(NSamples):
            sampleVec = priorSamples[ns]
            HParams = sampleToHamiltonianParameters(sampleVec, rvIndices, f2_bool)
            # simstart = timer()
            if simType == 'True':
                spectED_ds = trueSim(tgrid, spinBasis, HParams, shotNoiseParams, decayRate_true)
            elif simType == 'Trotter':
                spectED_ds = trotterSim_fixedGC_specOnly(tgrid, spinBasis, HParams, trotParamDict, shotNoiseParams, decayRate_trot)
            else:
                print('SIM TYPE ERROR')
            # simend = timer()
            # print(simend - simstart)
            spectrumSamples.append(spectED_ds['Spectrum'].values)
            hDSamples.append(hellingerDistance(spectED_ds['Spectrum'].values, Aw_true, fVals_true))
            hamSamples.append(HParams)

        hDmin_ind = np.argmin(np.array(hDSamples))
        Spectrum_da[iind + 1, :] = spectrumSamples[hDmin_ind]
        hellDist_da[iind + 1] = hDSamples[hDmin_ind]
        hamiltonian_da[iind + 1] = hamiltonianListToMatrix(hamSamples[hDmin_ind], spinBasis.L, f2_bool)
        hellDist_MeanOfSamples_da[iind + 1] = np.mean(np.array(hDSamples))
        hellDist_VarOfSamples_da[iind + 1] = np.var(np.array(hDSamples))

        HParams_mean = sampleToHamiltonianParameters(sampleMean, rvIndices, f2_bool)
        if simType == 'True':
            spectED_mean_ds = trueSim(tgrid, spinBasis, HParams_mean, shotNoiseParams, decayRate_true)
        elif simType == 'Trotter':
            spectED_mean_ds = trotterSim_fixedGC_specOnly(tgrid, spinBasis, HParams_mean, trotParamDict, shotNoiseParams, decayRate_trot)
        else:
            print('SIM TYPE ERROR')
        hellDist_mean_da[iind + 1] = hellingerDistance(spectED_mean_ds['Spectrum'].values, Aw_true, fVals_true)
        Spectrum_mean_da[iind + 1, :] = spectED_mean_ds['Spectrum'].values
        hamiltonian_mean_da[iind + 1] = hamiltonianListToMatrix(HParams_mean, spinBasis.L, f2_bool)

        print('Iteration: {0}, Hellinger Distance: {1}'.format(iind + 1, hellDist_mean_da[iind + 1].values))
        # if iind == 1:
        #     print('Second Hellinger Distance: {0}'.format(hellDist_mean_da[1].values))

        updateWeights = bayesianWeights(Aw_true_norm, fVals_true, spectrumSamples)

        if sampleMemory > 0:
            if np.isclose(sampleMemory_temp, 0):
                runningSamples = runningSamples[NSamples:]
                runningWeights = runningWeights[NSamples:]
            else:
                sampleMemory_temp -= 1
            runningSamples = [*runningSamples, *priorSamples]
            runningWeights = [*(sampleDiscount * np.array(runningWeights)), *updateWeights]
        else:
            runningSamples = priorSamples
            runningWeights = updateWeights

        runningSamplesMat = np.stack(runningSamples)
        runningWeights_Array = np.array(runningWeights)

        sampleMean = np.sum(np.transpose(runningSamplesMat * runningWeights_Array[:, np.newaxis]), axis=1) / np.sum(runningWeights_Array)
        sampleCov = np.cov(np.transpose(runningSamplesMat), aweights=runningWeights_Array)
        covNorm = np.sum(runningWeights_Array) - np.sum(runningWeights_Array**2) / np.sum(runningWeights_Array); covFix = covNorm / np.sum(runningWeights_Array)
        sampleCov = covFix * sampleCov
        sampleCov = covarianceRegularization(sampleCov, covRegParam)

        posteriorRV = multivariate_normal(mean=sampleMean, cov=sampleCov, allow_singular=True)
        posteriorSamples = posteriorRV.rvs(size=NSamples)

        # print(sampleMemory, sampleMemory_temp, -1 + len(runningSamples) / NSamples)
        print(sampleMean)
        print(np.sqrt(np.diag(sampleCov)))
        print(np.abs(np.sqrt(np.diag(sampleCov)) / sampleMean))
        print('\n')

    # print(hellDist_da.values)
    # print('\n')
    print(hellDist_mean_da.values)

    data_dict = {'Spectrum': Spectrum_da, 'HellingerDistance': hellDist_da, 'HamiltonianMatrix': hamiltonian_da, 'HamiltonianMatrix_mean': hamiltonian_mean_da, 'Spectrum_mean': Spectrum_mean_da, 'HellingerDistance_mean': hellDist_mean_da, 'HellingerDistance_MeanOfSamples': hellDist_MeanOfSamples_da, 'HellingerDistance_VarOfSamples': hellDist_VarOfSamples_da}
    coords_dict = {'f': fVals_true, 'iteration': np.arange(NIterations + 1), 'i': np.arange(spinBasis.L), 'j': np.arange(spinBasis.L)}
    attrs_dict = {'Nspins': spinBasis.N, 'c2': c2, 'gateCount': gateBudget, 'intGateTime': intGateTime, 'singRotTime': singRotTime, 'sampleMemory': sampleMemory, 'sampleDiscount': sampleDiscount, 'runAverage': trotParamDict['runAverage'], 'decayRate_true': decayRate_true, 'decayRate_trot': decayRate_trot, 'decayRate_trot_noHeating': decayRate_trot_noHeating}

    BI_ds = xr.Dataset(data_dict, coords=coords_dict, attrs=attrs_dict)
    return BI_ds
