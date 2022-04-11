"""
This file shows example of bode plot and impulse response generation from differential s-params
"""

import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp

#load in touchstone file containing s-params of 2m copper cable connector
#from: https://www.ieee802.org/3/ck/public/tools/cucable/mellitz_3ck_04_1119_CACR.zip
thru_file = "C:/Users/richa/Desktop/Thesis/mellitz_3ck_04_1119_CACR/Tp0_Tp5_28p5db_FQSFP_thru.s4p"
thru_network = rf.Network(thru_file)

port_def = np.array([[0, 1],[2, 3]])

#compute desired timestep for impulse response
nyquist_f = 26.56e9
symbol_t = 1/(2*nyquist_f)
samples_per_symbol = 64
t_d = symbol_t/samples_per_symbol

#load and source impedance
Zs = 50
Zl = 50

#compute differential transfer function and impulse response from s-params
H_thru, f, h_thru, t = sdp.four_port_to_diff(thru_network, port_def, Zs, Zl, option = 1, t_d = t_d)

#Plot transfer function of Channel
plt.figure(dpi = 1200)
plt.plot(1e-9*f,20*np.log10(abs(H_thru)), color = "blue", label = "THRU channel", linewidth = 0.8)
plt.ylabel('Mag. Response [dB]')
plt.xlabel('Frequency [GHz]')
plt.axvline(x=26.56,color = 'grey', label = "Nyquist Frequency")
plt.title("Bode Plot")
plt.grid()
plt.legend()

#visualize pulse response
pulse_response = sp.signal.fftconvolve(h_thru, np.ones(samples_per_symbol), mode = "same")
sdp.channel_coefficients(pulse_response, t, samples_per_symbol, 3, 20)

#%% crop impulse response for and save
plt.figure(dpi = 1200)
h_thru_crop = h_thru[44500:47500]
plt.plot(h_thru_crop)

np.save("./data/h_thru.npy",h_thru_crop)
np.save("./data/f.npy",f)
np.save("./data/TF_thru.npy",H_thru)