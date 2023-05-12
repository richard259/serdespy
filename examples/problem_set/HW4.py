import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp

f = np.load("./data/f.npy")
w = np.load("./data/w.npy")

h_pulse = np.load("./data/hpulse.npy")
t = np.load("./data/t.npy")

signal = np.load("./data/signal.npy")
#Hchannel = np.load("./data/Hchannel.npy")

samples_per_symbol = 40
data_rate = 100e9

#set poles and zeroes for peaking at nyquist freq
#high peaking because channel is high insertion loss
z = 5e10
p = 1.7e11
k = p**2/z

#calculate Frequency response of CTLE at given frequencies
w, H_ctle = sp.signal.freqs([k/p**2, k*z/p**2], [1/p**2, 2/p, 1], w)

#bode plot of CTLE transfer function
plt.figure(dpi=600)
plt.semilogx(1e-9*f,20*np.log10(abs(H_ctle)), color = "red", label = 'CTLE')
plt.title("CTLE Frequency Response")
plt.grid()
plt.axvline(x=25,color = 'grey', label = "Nyquist Frequency")
plt.axvline(x=z/(2*np.pi)*1e-9,color = 'green', label = "Zero Location")
plt.axvline(x=p/(2*np.pi)*1e-9,color = 'blue', label = "Pole Location")
plt.legend()

#%%
h_ctle, t_ctle = sdp.freq2impulse(H_ctle,f)
h_ctle = h_ctle[0:200]
plt.figure(dpi=600)
plt.plot(h_ctle)

#%% Eye diagram of signal with and without CTLE

sdp.simple_eye(signal[100*samples_per_symbol:], samples_per_symbol*3, 500, 500*1e-15, "{}Gbps 4-PAM Signal".format(data_rate/1e9))

signal_ctle = sp.signal.convolve(signal,h_ctle)

sdp.simple_eye(signal_ctle[100*samples_per_symbol:], samples_per_symbol*3, 500, 500*1e-15, "{}Gbps 4-PAM Signal with CTLE".format(data_rate/1e9))

#%%
h_pulse_ctle = sp.signal.convolve(h_pulse,h_ctle)

FFE_pre = 2
FFE_taps = 7
FFE_post = FFE_taps - FFE_pre - 1
DFE_taps = 2

sdp.channel_coefficients(h_pulse[:t.size],t,40,2,4)

h = sdp.channel_coefficients(h_pulse_ctle[:t.size],t,40,2,4)
#%%
#h /= h.max()
#print('h: ',h)

channel_main = h.argmax()

#main_cursor = h[channel_main]
main_cursor = 1

#generate binary data
data = sdp.prqs10(1)[:10000]

voltage_levels = np.array([-3, -1, 1, 3])

#generate Baud-Rate sampled signal from data
signal_BR = sdp.pam4_input_BR(data)

signal_rx = sp.signal.fftconvolve(h, signal_BR)[:len(signal_BR)]

signal_rx_cropped = signal_rx[channel_main:]

reference_signal = signal_BR[:1000]

w_ffe_init = np.zeros([7,])
w_dfe_init = np.zeros([2,])

w_ffe, w_dfe, v_combined_ffe, v_combined_dfe, z_combined, e_combined = \
sdp.lms_equalizer(signal_rx_cropped, 0.001, len(signal_rx_cropped), w_ffe_init, FFE_pre, w_dfe_init,  voltage_levels, reference=reference_signal[:1000])

#%%

#voltage_levels = np.array([-3,-1,1,3])

nyquist_f = 25e9

RX = sdp.Receiver(signal_ctle, samples_per_symbol, nyquist_f, voltage_levels,main_cursor=main_cursor)

#sdp.simple_eye(RX.signal, samples_per_symbol*3, 800, TX.UI/TX.samples_per_symbol, "Eye Diagram with CTLE")

RX.FFE(w_ffe, FFE_pre)

sdp.simple_eye(RX.signal[int(100.5*samples_per_symbol):], samples_per_symbol*3, 800, 500*1e-15, "Eye Diagram with CTLE and FFE")

RX.pam4_DFE(w_dfe)

sdp.simple_eye(RX.signal[int(100.5*samples_per_symbol):], samples_per_symbol*3, 800, 500*1e-15, "Eye Diagram with CTLE, FFE, and DFE")