"""
This file shows example of generating response of crosstalk on the channel

Running this file is optional, you can skip to 5_BER_test.py, but if you skip this file make sure to comment lines 50 and 51 of 5_BER_test.py

to run this file, it is necessary to have downloaded the entire folder of s-params given in the header of 1_channel.py

write the path to the unzipped directory below, on line 18
"""

import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp

#path to touchstone files here:
s4p_dir = "C:/Users/richa/Downloads/mellitz_3ck_04_1119_CACR/"

#far-end crosstalk
fext_files = ['Tp0_Tp5_28p5db_FQSFP_fext' + f'{i}' +'.s4p' for i in range(1,8)]
fext_networks = [rf.Network(s4p_dir + fext_file ) for fext_file in fext_files]

#near-end crosstalk
next_files = ['Tp0_Tp5_28p5db_FQSFP_next' + f'{i}' +'.s4p'  for i in range(1,9)]
next_networks = [rf.Network(s4p_dir + next_file ) for next_file in next_files]

#set-up params
port_def = np.array([[0, 1],[2, 3]])

nyquist_f = 26.56e9
symbol_t = 1/(2*nyquist_f)
samples_per_symbol = 64
t_d = symbol_t/samples_per_symbol

#%%Compute response for each of the crosstalk agressors at the near and far ends

H_fext = np.zeros((len(fext_files), 169985), dtype = complex)
h_fext = np.zeros((len(fext_files), 339968))

for i in range(len(fext_files)):
    H, f, h, t = sdp.four_port_to_diff(fext_networks[i], port_def, 50, np.inf, option = 0, t_d = t_d)
    H_fext[i] = H
    h_fext[i] = h

H_next = np.zeros((len(next_files), 169985), dtype = complex)
h_next = np.zeros((len(next_files), 339968))

for i in range(len(next_files)):
    H, f, h, t = sdp.four_port_to_diff(next_networks[i], port_def, 50, np.inf, option = 0, t_d = t_d)
    H_next[i] = H
    h_next[i] = h

#%%% Plot magnitude response of crosstalk and thru channel
import matplotlib.patches as mpatches

plt.figure(dpi = 1200)

next_patch = mpatches.Patch(color='red', label='FEXT')
fext_patch = mpatches.Patch(color='orange', label='NEXT')
thru_patch = mpatches.Patch(color='blue', label='THRU')
nyquist_patch = mpatches.Patch(color='grey', label='Nyquist Frequency')

for i in range(len(fext_files)):
    plt.plot(1e-9*f,20*np.log10(abs(H_fext[i,:])), color = "red", linewidth = 0.2)
    
for i in range(len(next_files)):
    plt.plot(1e-9*f,20*np.log10(abs(H_next[i,:])), color = "orange", linewidth = 0.2)
    
H_thru = np.load("./data/TF_thru.npy")
plt.plot(1e-9*f,20*np.log10(abs(H_thru)), color = "blue", label = "THRU channel", linewidth = 0.8)

plt.xlim([0,60])
plt.ylim([-125,5])
plt.ylabel('Mag. Response [dB]')
plt.xlabel('Frequency [GHz]')
plt.axvline(x=26.56,color = 'grey', label = "Nyquist Frequency")
plt.title("Channel and Crosstalk Bode Plot")
plt.grid()
plt.legend(loc = "upper right", handles = [next_patch, fext_patch, thru_patch, nyquist_patch])
#plt.savefig(fig_dir + "12dB_bode", transparent = True)

#%%Generate crosstalk response to random data

h_xtalk = np.vstack((h_next, h_fext))

voltage_levels = np.array([-3,-1,1,3])

data = sdp.prqs10(1)

TX = sdp.Transmitter(data[:10000], voltage_levels, 26.56e9)

TX.oversample(samples_per_symbol)

xt_response = np.zeros([data.size*samples_per_symbol,])

for i in range(h_xtalk.shape[0]):
    print(i)
    
    #generate data for each xtalk channel with new random seed
    data = sdp.prqs10(int(i+1))
    
    #find xtalk response and sum
    TX = sdp.Transmitter(data, voltage_levels, 26.56e9)
    TX.oversample(samples_per_symbol)
    xt_response = xt_response + sp.signal.fftconvolve(TX.signal_ideal, h_xtalk[i][:], mode = "same")

#plot eye diagram of sum of all crosstalk
sdp.simple_eye(xt_response, samples_per_symbol*3, 500, TX.UI/TX.samples_per_symbol, "XT")

#save data
np.save("./data/xt_response.npy",xt_response)


