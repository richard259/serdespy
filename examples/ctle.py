"""
This file shows example of CTLE model
"""
import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp

f = np.load("./data/f.npy")
h = np.load("./data/h_thru.npy")
H = np.load("./data/TF_thru.npy")

samples_per_symbol = 64

#set poles and zeroes for peaking at nyquist freq
#high peaking because channel is high insertion loss
z = 8.5e8
p = 4.2e9

#compute transfer function coefficients for 2-zero 4-pole system
b,a = sp.signal.zpk2tf([z, z],[p, p, p, p],1)
b *= 1/(b[-1]/a[-1])

#frequency vector in rad/s
w = f/(2*np.pi)

#calculate Frequency response of CTLE at given frequencies
w, H_ctle = sp.signal.freqs(b, a, w)

#bode plot of CTLE transfer function
plt.figure(dpi=1200)
plt.semilogx(1e-9*f,20*np.log10(abs(H_ctle)), color = "red", label = "ctle")
plt.ylabel('Mag. Response [dB]')
plt.xlabel('Frequency [GHz]')
plt.title("CTLE Bode Plot")
plt.grid()
plt.axvline(x=26.56,color = 'grey', label = "Nyquist Frequency")

#%% compute and save impulse response of CTLE transfer function

h_ctle, t_ctle = sdp.freq2impulse(H_ctle,f)

crop = 200

h_ctle = np.flip(h_ctle[-crop:])
plt.figure(dpi=1200)
plt.plot(h_ctle)

np.save("./data/h_ctle.npy", h_ctle)

#%% plot eye diagram with and without CTLE
voltage_levels = np.array([-3,-1,1,3])

nyquist_f = 26.56e9

data = sdp.prqs10(1)

TX = sdp.Transmitter(data[:10000], voltage_levels, nyquist_f)

TX.oversample(samples_per_symbol)

signal_out = sp.signal.fftconvolve(TX.signal_ideal, h, mode = "same")[:64*1000*5]

sdp.simple_eye(signal_out[1000:], samples_per_symbol*3, 1000, TX.UI/TX.samples_per_symbol, "Eye Diagram")

signal_out_ctle = sp.signal.fftconvolve(signal_out, h_ctle, mode = "same")

sdp.simple_eye(signal_out_ctle[1000:], samples_per_symbol*3, 1000, TX.UI/TX.samples_per_symbol, "Eye Diagram with CTLE")
