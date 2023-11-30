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
  datapath = '/Users/kis/Desktop/'


  # # # ---- Chemistry inspired simulation ----

  N = 4  # Number of qubits indexed i=0,...,N-1 with 0 corresponding to the ancilla qubit

  JijList_desired = [[100, 1, 2], [115, 2, 3]]  # Desired Heisenberg interaction strengths (MHz) in form [Jij, i, j]
  JijList_sc = []  # Required flip-flop interaction strengths (MHz) between physical qubits. Note that we ignore parasitic ZZ interactions (~0.2 MHz each).
  for tup in JijList_desired:
    [val, i, j] = tup
    JijList_sc.append([3 * val / 2, i, j])
  hiList_sc = [[0.0, 0], [0.4, 1], [0.7, 2], [1.1, 3]]  # physical qubit frequencies (MHz) in rotating frame

  tscale = 1 / 100  # time scale (microseconds) of effective dynamics
  tmax = 10 * tscale; dt = tscale/10
  tgrid = np.arange(0, tmax, dt)  # time grid (microseconds)
  print(tscale, np.max(tgrid), dt)

  # Noise parameters

  T1_exp = 15  # T1 time (microseconds)
  T2_exp = 3   # T2 time (microseconds)

  kappa = 10  # sets the scale of dissipation as 10 MHz, with the smalled decoherence time implementable by the channel being T = 1/kappa = 0.1 microseconds
  gamma_amp = 1/(kappa**2 * T1_exp)  # corresponds to a qubit occupation decay of exp(-t/T1) with T1 = 1/(kappa**2 * gamma_amp) where gamma \in [0, 1]
  gamma_phase = 1/(2*kappa**2 * T2_exp)  # corresponds to a qubit coherence decay of 0.5 + 0.5 * exp(-t/T2) with T2 = 1/(2*kappa**2 * gamma_phase) where gamma \in [0, 1]

  # # Checking noise parameters

  # tgrid_noisefit = np.arange(0,45,0.1)

  # T1_ds_amp = scfuncs.T1sim(tgrid_noisefit, kappa, gamma_amp, 0) 
  # T2_ds_amp = scfuncs.T2sim(tgrid_noisefit, kappa, gamma_amp, 0) 

  # fig_T1, ax_T1 = plt.subplots()
  # ax_T1.plot(tgrid_noisefit, T1_ds_amp['state_prob'].values,'k-')
  # ax_T1.plot(tgrid_noisefit, T2_ds_amp['state_prob'].values,'g-')
  # ax_T1.plot(tgrid_noisefit, np.exp(-1*tgrid_noisefit/T1_exp),'r--')

  # T1_ds_phase = scfuncs.T1sim(tgrid_noisefit, kappa, 0, gamma_phase) 
  # T2_ds_phase = scfuncs.T2sim(tgrid_noisefit, kappa, 0, gamma_phase) 

  # fig_T2, ax_T2 = plt.subplots()
  # ax_T2.plot(tgrid_noisefit, T1_ds_phase['state_prob'].values,'k-')
  # ax_T2.plot(tgrid_noisefit, T2_ds_phase['state_prob'].values,'g-')
  # ax_T2.plot(tgrid_noisefit, 0.5 + 0.5 * np.exp(-1*tgrid_noisefit/T2_exp),'r--')

  # plt.show()

  # Simulation

  # operator_list = scfuncs.getSpinOperators(N)
  # Hchem = scfuncs.makeGenerator_Hchem(operator_list, JijList_desired)
  # Hsc = scfuncs.makeGenerator_Hsc(operator_list, JijList_sc, hiList_sc)

  tstart = timer()
  Hchem_ds = scfuncs.chemSim_qT(tgrid, N, JijList_desired, kappa, gamma_amp, gamma_phase)
  print(timer() - tstart)


