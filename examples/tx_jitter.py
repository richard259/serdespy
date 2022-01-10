"""Example of tx_jitter with NRZ and PAM-4 input"""

import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#NRZ
voltage_levels = np.array([-1,1])
PRBS = sdp.prbs13(1)
samples_per_symbol=128
ideal = sdp.nrz_input(samples_per_symbol,PRBS,voltage_levels)
nyquist_f = 26.56e9
nyquist_T = 1/nyquist_f
UI = nyquist_T/2
superposition,non_ideal = sdp.tx_jitter(UI,samples_per_symbol,ideal)
sdp.simple_eye(superposition,384,500,UI,"tx jitter eye diagram - NRZ")
plt.show()

plt.plot(superposition[:1000],'b') #plots waveform of TX jittered output
plt.plot(non_ideal[:1000],'r') #plots TX jitter waveform
plt.plot(ideal[:1000],color='g') #plots ideal input waveform
plt.show()
#%%

#PAM4
voltage_levels = np.array([-3,-1,1,3])
PRQS = sdp.prqs10(1)
samples_per_symbol=128
ideal = sdp.pam4_input(samples_per_symbol,PRQS,voltage_levels)
nyquist_f = 26.56e9
nyquist_T = 1/nyquist_f
UI = nyquist_T/2
superposition,non_ideal = sdp.tx_jitter(UI,samples_per_symbol,ideal)
sdp.simple_eye(superposition,384,500,UI,"tx jitter eye diagram - PAM4")
plt.show()

plt.plot(superposition[:1000],'b') #plots waveform of TX jittered output
plt.plot(non_ideal[:1000],'r') #plots TX jitter waveform
plt.plot(ideal[:1000],color='g') #plots ideal input waveform
plt.show()