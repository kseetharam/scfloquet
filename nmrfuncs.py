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


def makeGenerators_sc(JijList, hiList, N, gamma, decohType):

    JijList_sc, hiList_sc = scHamParams(JijList, hiList)

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

    for tup in hiList_sc:
        hi, i = tup
        if hi != 0:
            H += hi * sz_list[i]

    for tup in JijList_sc:
        Jij, i, j = tup
        H += Jij * (sx_list[i] * sx_list[j] + sy_list[i] * sy_list[j])

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


def makeRotations(N, pulseAngle):
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

    sx_tot = 0; sy_tot = 0; sz_tot = 0
    for n in range(N):
        sx_tot += sx_list[n]
        sy_tot += sy_list[n]
        sz_tot += sz_list[n]

    Rx = (-1j*pulseAngle*sx_tot).expm()
    Ry = (-1j*pulseAngle*sy_tot).expm()
    Rz = (-1j*pulseAngle*sz_tot).expm()    

    return Rx, Ry, Rz



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


def trueSim_complex_qT(tgrid, spinBasis, HParams, gamma, decohType):
    [JijList, hiList] = HParams
    N = spinBasis.L

    pulseAngle = np.pi / 2

    Rx, Ry, Rz = makeRotations(N,pulseAngle)
    Rx_m, Ry_m, Rz_m = makeRotations(N,-1*pulseAngle)
    sRx = qt.to_super(Rx); sRy = qt.to_super(Ry); sRz = qt.to_super(Rz)
    sRx_m = qt.to_super(Rx_m); sRy_m = qt.to_super(Ry_m); sRz_m = qt.to_super(Rz_m)

    H, c_op_list = makeGenerators(JijList, hiList, N, gamma, decohType)

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

def trueSim_complex_qT_aveHam(tgrid, spinBasis, HParams, gamma, decohType):
    [JijList, hiList] = HParams
    N = spinBasis.L

    pulseAngle = np.pi / 2

    Rx, Ry, Rz = makeRotations(N,pulseAngle)
    Rx_m, Ry_m, Rz_m = makeRotations(N,-1*pulseAngle)
    sRx = qt.to_super(Rx); sRy = qt.to_super(Ry); sRz = qt.to_super(Rz)
    sRx_m = qt.to_super(Rx_m); sRy_m = qt.to_super(Ry_m); sRz_m = qt.to_super(Rz_m)
    H, c_op_list = makeGenerators_sc(JijList, hiList, N, gamma, decohType)

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
