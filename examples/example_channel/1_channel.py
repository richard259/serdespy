"""
This file shows example of loading in touchstone file for differential channel and generating bode plot and impulse response

To run this file:
    - download zipped touchstone files from : https://www.ieee802.org/3/ck/public/tools/cucable/mellitz_3ck_04_1119_CACR.zip
    - Place the file Tp0_Tp5_28p5db_FQSFP_thru.s4p in the working directory, with this file. It contains s-paramater measurements for 2m copper cable connector with 28dB insertion loss at 26.56Ghz from IEEE 802.df public channels
    - create a subdirectory called 'data' in the working directory. this is where data generated from this script will be saved for use in other examples
"""

#import packages
import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp

#load in touchstone file containing s-params of 2m copper cable connector
thru_file = "./Tp0_Tp5_28p5db_FQSFP_thru.s4p"
thru_network = rf.Network(thru_file)

#port definition, is defined in the header of the touchstone file
port_def = np.array([[0, 1],[2, 3]])

#for pam-4 signalling at 106.24 Gb/s
nyquist_f = 26.56e9

#time per pam-4 symbol
symbol_t = 1/(2*nyquist_f)

#oversampling ratio is 64 samples per 4-pam symbol, to get smooth eye diagrams
samples_per_symbol = 64

#compute desired timestep for impulse response
t_d = symbol_t/samples_per_symbol

#load and source impedance are matched 50 ohms, because charictaristic empedance of the the channel is 50 ohms
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

#crop impulse response and save
plt.figure(dpi = 1200)
h_thru_crop = h_thru[44500:47500]
plt.plot(h_thru_crop)

#save pulse response, transfer function, and frequency vector, used in other example files

#save data
np.save("./data/h_thru.npy",h_thru_crop)
np.save("./data/f.npy",f)
np.save("./data/TF_thru.npy",H_thru)