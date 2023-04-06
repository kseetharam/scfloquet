import numpy as np
import scfuncs
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
  lw = 0.5

  # # # ---- NMR simulation ----

  magnet = 16.1
  offset_Hz = 2800
  sweep_Hz = 700
  gamma_1H = 2.6752 * 1e8
  basefrq = -gamma_1H * magnet / (2 * np.pi)
  npoints = 1024

  timestep = 1 / sweep_Hz
  tgrid = timestep * np.array([i for i in range(npoints)])
  fgrid = np.fft.fftshift(np.fft.fftfreq(npoints, timestep))

  cshifts_ppm = np.array([3.70, 3.92, 4.50])  # in ppm
  cshifts_Hz = basefrq * cshifts_ppm * 1e-6

  Jij_Hz = [[cshifts_Hz[0], 10, 4],
            [10, cshifts_Hz[1], 12],
            [4, 12, cshifts_Hz[2]]]

  Jij_Hz_offset = np.array(Jij_Hz) + np.eye(len(Jij_Hz)) * offset_Hz

  hamMat_radHz = 2 * np.pi * Jij_Hz_offset
  # frameshift = np.exp(-1j * 2 * np.pi * offset_Hz * tgrid)

  fgrid_offset = fgrid + offset_Hz
  fgrid_ppm = -1e6 * fgrid_offset / basefrq

  N = hamMat_radHz.shape[0]
  hiList = [[hamMat_radHz[i, i], i] for i in np.arange(N)]  # extracts hi from parameter matrix (puts in form for QuSpin)
  JijList = [[hamMat_radHz[i, j], i, j] for i in np.arange(N) for j in np.arange(N) if (i != j) and (i < j) if not np.isclose(hamMat_radHz[i, j], 0)]  # extracts Jij from parameter matrix (puts in form for QuSpin); this list combines the Jij and Jji terms (Hermitian conjugates) into a single term
  spinBasis = spin_basis_1d(N, pauli=False)
  HParams = [JijList, hiList]
  gridParams = {'spinBasis': spinBasis, 'tgrid': tgrid}
  shotNoiseParams_true = {'ShotNoise': False, 'N_ShotNoise': 10000}

  print(timestep)

  # Noise parameters

  # Note: for the superconducting simulation, we will assume the Hamiltonian parameters above (given in Hz) are actually values in MHz and the time points (given in seconds) are in microseconds.  
  # This is equivalent to rescaling H->1e6*H and t->1e-6*t so that H*t is invariant. Therefore, for the purposes of setting the dissipation rate, we interpret tgrid as values in microseconds.

  T1_exp = 20  # T1 time in microseconds
  T2_exp = 3   # T2 time in microseconds

  kappa = 10  # sets the scale of dissipation as 10 MHz, with the smalled decoherence time implementable by the channel being T = 1/kappa = 0.1 microseconds
  gamma_amp = 1/(kappa**2 * T1_exp)  # corresponds to a qubit occupation decay of exp(-t/T1) with T1 = 1/(kappa**2 * gamma_amp) where gamma \in [0, 1]
  gamma_phase = 1/(2*kappa**2 * T2_exp)  # corresponds to a qubit coherence decay of 0.5 + 0.5 * exp(-t/T2) with T2 = 1/(2*kappa**2 * gamma_phase) where gamma \in [0, 1]
  apo_param = 4

  # # Fitting noise parameters

  # tgrid_noisefit = timestep * np.array([i for i in range(10*npoints)])

  # T1_ds_phase = scfuncs.T1sim(tgrid_noisefit, kappa, 0, gamma_phase) 
  # T2_ds_phase = scfuncs.T2sim(tgrid_noisefit, kappa, 0, gamma_phase) 

  # fig2, ax2 = plt.subplots()
  # ax2.plot(tgrid_noisefit, T1_ds_phase['state_prob'].values,'k-')
  # ax2.plot(tgrid_noisefit, T2_ds_phase['state_prob'].values,'g-')
  # ax2.plot(tgrid_noisefit, 0.5 + 0.5 * np.exp(-1*tgrid_noisefit/T2_exp),'r--')

  # T1_ds_amp = scfuncs.T1sim(tgrid_noisefit, kappa, gamma_amp, 0) 
  # T2_ds_amp = scfuncs.T2sim(tgrid_noisefit, kappa, gamma_amp, 0) 

  # fig2, ax2 = plt.subplots()
  # ax2.plot(tgrid_noisefit, T1_ds_amp['state_prob'].values,'k-')
  # ax2.plot(tgrid_noisefit, T2_ds_amp['state_prob'].values,'g-')
  # ax2.plot(tgrid_noisefit, np.exp(-1*tgrid_noisefit/T1_exp),'r--')

  # plt.show()

  # Noiseless simulation (true Hamiltonian)

  tstart = timer()
  trueED_ds = scfuncs.trueSim_complex(tgrid, spinBasis, HParams, shotNoiseParams_true, 1)
  print(timer() - tstart)
  fid_raw = trueED_ds['ResponseFunc_Real'].values + 1j * trueED_ds['ResponseFunc_Imag'].values
  fid_apo = scfuncs.apodize_exp1d(fid_raw - np.mean(fid_raw), apo_param)
  spec_noiseless = np.real(np.fft.fftshift(np.fft.fft(fid_apo)))

  # # Noiseless simulation (average Hamiltonian)

  # tstart = timer()
  # trueED_ds = scfuncs.trueSim_complex_aveHam(tgrid, spinBasis, HParams, shotNoiseParams_true, 1)
  # print(timer() - tstart)
  # fid_raw = trueED_ds['ResponseFunc_Real'].values + 1j * trueED_ds['ResponseFunc_Imag'].values
  # fid_apo = scfuncs.apodize_exp1d(fid_raw - np.mean(fid_raw), apo_param)
  # spec_noiseless_aveHam = np.real(np.fft.fftshift(np.fft.fft(fid_apo)))

  # # Noisy simulation (true Hamiltonian)

  # tstart = timer()
  # trueED_ds = scfuncs.trueSim_complex_qT(tgrid, spinBasis, HParams, kappa, gamma_amp, gamma_phase)
  # print(timer() - tstart)
  # fid_raw = trueED_ds['ResponseFunc_Real'].values + 1j * trueED_ds['ResponseFunc_Imag'].values
  # fid_apo = scfuncs.apodize_exp1d(fid_raw - np.mean(fid_raw), apo_param)
  # spec_noisy = np.real(np.fft.fftshift(np.fft.fft(fid_apo)))

  # Noisy simulation (average Hamiltonian)

  tstart = timer()
  trueED_ds = scfuncs.trueSim_complex_qT_aveHam(tgrid, spinBasis, HParams, kappa, gamma_amp, gamma_phase)
  print(timer() - tstart)
  fid_raw = trueED_ds['ResponseFunc_Real'].values + 1j * trueED_ds['ResponseFunc_Imag'].values
  fid_apo = scfuncs.apodize_exp1d(fid_raw - np.mean(fid_raw), apo_param)
  spec_noisy_aveHam = np.real(np.fft.fftshift(np.fft.fft(fid_apo)))


  # Plot 

  fig, ax = plt.subplots()
  ax.plot(fgrid, spec_noiseless,linewidth=lw,color='k',linestyle='-')
  # ax.plot(fgrid, spec_noiseless_aveHam,linewidth=lw,color='g',linestyle='-')
  # ax.plot(fgrid, spec_noisy,linewidth=lw,color='b',linestyle='-')
  ax.plot(fgrid, spec_noisy_aveHam,linewidth=lw,color='r',linestyle='-')
  ax.set_ylim([0, 20])
  # ax.set_xlim([-300, -200])

  plt.show()
