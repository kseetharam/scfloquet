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

  # # # ---- XXZ simulation ----

  N = 4  # number of qubits. If N < 4, assumes a line of qubits. If N > 4, assumes a square grid of qubits of size sqrt(N) x sqrt(N)

  hiList = [0 for i  in np.arange(N**2)]  # qubit frequencies in MHz
  J_I = 0.2  # parasitic ZZ interaction strength of native SC Hamiltonian (MHz)
  J_S = 6  # flip-flop interaction strength of native SC Hamiltonian (MHz)
  Delta_eff = 0.3  # anisotropy of effective XXZ Hamiltonian (in units of J_eff); leads to well-defined pulse times for values between ~[0.1, 2]
  alpha = (J_I * Delta_eff + J_S * (Delta_eff - 2))/(J_I - J_S * Delta_eff)  # pulse anisotropy parameter (tau_z = alpha * tau)
  J_eff = (J_I + (alpha + 1) * J_S)/(alpha + 2) # flip-flop interaction strength of effective XXZ Hamiltonian (MHz)
  hi_eff_List = [(alpha / (alpha + 2)) * hi for hi in hiList]

  tscale = 1 / J_eff  # time scale of effective dynamics in microseconds
  tmax = 10 * tscale; dt = tscale/10
  tgrid = np.arange(0, tmax, dt)  # time grid in microseconds
  print(tscale, np.max(tgrid), dt)

  operator_list = scfuncs.getSpinOperators(N)
  H = scfuncs.makeGenerator_Hnative(operator_list, J_S, J_I, hiList)

  # Noise parameters

  T1_exp = 20  # T1 time in microseconds
  T2_exp = 1   # T2 time in microseconds

  kappa = 10  # sets the scale of dissipation as 10 MHz, with the smalled decoherence time implementable by the channel being T = 1/kappa = 0.1 microseconds
  gamma_amp = 1/(kappa**2 * T1_exp)  # corresponds to a qubit occupation decay of exp(-t/T1) with T1 = 1/(kappa**2 * gamma_amp) where gamma \in [0, 1]
  gamma_phase = 1/(2*kappa**2 * T2_exp)  # corresponds to a qubit coherence decay of 0.5 + 0.5 * exp(-t/T2) with T2 = 1/(2*kappa**2 * gamma_phase) where gamma \in [0, 1]

  # # Pair test

  # tstart = timer()
  # # ds = scfuncs.xxzSim_pair_qT(tgrid, J_eff, Delta_eff, hiList, kappa, gamma_amp, gamma_phase)
  # ds = scfuncs.xxzSim_pair_qT(tgrid, J_eff, Delta_eff, hi_eff_List, kappa, 0, 0)
  # print(timer() - tstart)
  # state_prob = ds['state_prob'].values
  # Sy = ds['Sy'].values

  # tstart = timer()
  # # ds_aveHam = scfuncs.xxzSim_pair_qT_aveHam(tgrid, J_S, J_I, hiList, Delta_eff, alpha, kappa, gamma_amp, gamma_phase)
  # ds_aveHam = scfuncs.xxzSim_pair_qT_aveHam(tgrid, J_S, J_I, hiList, Delta_eff, alpha, kappa, 0, 0)
  # print(timer() - tstart)
  # state_prob_aveHam = ds_aveHam['state_prob'].values
  # Sy_aveHam = ds_aveHam['Sy'].values

  # fig, ax = plt.subplots()
  # ax.plot(tgrid, state_prob,linewidth=lw,color='b',linestyle='-')
  # ax.plot(tgrid, state_prob_aveHam,linewidth=lw,color='r',linestyle='--')
  # ax.plot(tgrid, Sy,linewidth=lw,color='k',linestyle='-')
  # ax.plot(tgrid, Sy_aveHam,linewidth=lw,color='g',linestyle='--')
  # ax.set_ylim([-1.1, 1.1])
  # plt.show()

  # Grid test

  tstart = timer()
  ds = scfuncs.xxzSim_qT(tgrid, N, J_eff, Delta_eff, hiList, kappa, gamma_amp, gamma_phase)
  # ds = scfuncs.xxzSim_qT(tgrid, N, J_eff, Delta_eff, hi_eff_List, kappa, 0, 0)
  print(timer() - tstart)
  Sy = ds['Sy'].values

  tstart = timer()
  ds_aveHam = scfuncs.xxzSim_qT_aveHam(tgrid, N, J_S, J_I, hiList, Delta_eff, alpha, kappa, gamma_amp, gamma_phase)
  # ds_aveHam = scfuncs.xxzSim_qT_aveHam(tgrid, N, J_S, J_I, hiList, Delta_eff, alpha, kappa, 0, 0)
  print(timer() - tstart)
  Sy_aveHam = ds_aveHam['Sy'].values

  fig, ax = plt.subplots()
  ax.plot(tgrid, Sy,linewidth=lw,color='k',linestyle='-')
  ax.plot(tgrid, Sy_aveHam,linewidth=lw,color='r',linestyle='--')
  # ax.set_ylim([-1.1, 1.1])
  plt.show()
