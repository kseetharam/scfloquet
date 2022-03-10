import numpy as np
import nmrfuncs as nfuncs
import compSense as cs
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from scipy.io import savemat, loadmat
from copy import copy, deepcopy
import os
import sys
import csv
from timeit import default_timer as timer
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import qutip as qt

if __name__ == "__main__":

  np.set_printoptions(linewidth=2000)
  matplotlib.rcParams.update({'font.size': 12})
  labelsize = 13
  legendsize = 12

  # # # ---- Acetonitrile ZF dissipation test ----

  lowfield = False

  spinach_datapath = '/Users/kis/Dropbox/NMR Learning/Code/matlab/spectrumSim/fid_data/'

  sweep_Hz = 700; npoints = 4096; timestep = 1 / sweep_Hz
  gamma_1H = 2.6752 * 1e8; gamma_13C = 0.6728 * 1e8
  weights = np.array([gamma_1H, gamma_1H, gamma_1H, gamma_13C]) / gamma_1H

  timestep = 1 / sweep_Hz
  tgrid = timestep * np.array([i for i in range(npoints)])
  # tgrid = (timestep / 2) * np.array([i for i in range(2 * npoints)])
  fgrid = np.fft.fftshift(np.fft.fftfreq(npoints, timestep))

  Jij_Hz_zf = np.array([[0, 0, 0, 136.2],
                        [0, 0, 0, 136.2],
                        [0, 0, 0, 136.2],
                        [136.2, 136.2, 136.2, 0]])

  Jij_Hz_ulf = np.array([[-11.2405, 0, 0, 136.2],
                         [0, -11.2405, 0, 136.2],
                         [0, 0, -11.2405, 136.2],
                         [136.2, 136.2, 136.2, -2.827]])

  if lowfield:
    Jij_Hz = Jij_Hz_ulf
    nstring = 'ulf'
  else:
    Jij_Hz = Jij_Hz_zf
    nstring = 'zf'

  hamMat_radHz = 2 * np.pi * Jij_Hz

  N = hamMat_radHz.shape[0]
  hiList = [[hamMat_radHz[i, i], i] for i in np.arange(N)]  # extracts hi from parameter matrix (puts in form for QuSpin)
  JijList = [[hamMat_radHz[i, j], i, j] for i in np.arange(N) for j in np.arange(N) if (i != j) and (i < j) if not np.isclose(hamMat_radHz[i, j], 0)]  # extracts Jij from parameter matrix (puts in form for QuSpin); this list combines the Jij and Jji terms (Hermitian conjugates) into a single term
  spinBasis = spin_basis_1d(N, pauli=False)
  HParams = [JijList, hiList]
  gridParams = {'spinBasis': spinBasis, 'tgrid': tgrid}
  shotNoiseParams_true = {'ShotNoise': False, 'N_ShotNoise': 10000}

  gamma = 0.25
  # decohType = 'individual'
  decohType = 'symmetric'

  # H, c_op_list = nfuncs.makeGenerators(JijList, hiList, N, gamma, decohType)
  # psi0 = qt.tensor([qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0)])
  # result = qt.mesolve(H, psi0, tgrid, c_op_list)
  # print(np.abs(np.diag(result.states[0].full())))

  tstart = timer()
  trueED_ds = nfuncs.trueSim_zerofield_qT(tgrid, spinBasis, HParams, weights, gamma, decohType)
  # trueED_ds = nfuncs.trueSim_zerofield(tgrid, spinBasis, HParams, weights, shotNoiseParams_true, lowfield)
  print(timer() - tstart)
  fid_raw = trueED_ds['ResponseFunc'].values

  # entEntropy = trueED_ds['EntEntropy'].values
  # np.savetxt('/Users/kis/Downloads/entEntropy_acetonitrile_zf', entEntropy)

  # overlap = trueED_ds['overlap'].values
  # overlap_Ave = trueED_ds['overlap_Ave'].values
  # np.savetxt('/Users/kis/Downloads/overlap_acetonitrile_zf', overlap)
  # savemat('/Users/kis/Downloads/overlap_acetonitrile_zf.mat', {'overlap': overlap})
  # np.savetxt('/Users/kis/Downloads/overlap_Ave_acetonitrile_zf', overlap_Ave)

  # plt.plot(tgrid, fid_raw)

  fig, ax = plt.subplots()
  fid_apo = nfuncs.apodize_exp1d(fid_raw - np.mean(fid_raw), 12)
  spec_apo = np.real(np.fft.fftshift(np.fft.fft(fid_apo)))
  ax.set_xlim([-10, 350])
  ax.plot(fgrid, spec_apo)

  # fig2, ax2 = plt.subplots()
  # ax2.plot(tgrid, fid_raw)
  # # ax2.plot(tgrid, entEntropy)

  plt.show()
