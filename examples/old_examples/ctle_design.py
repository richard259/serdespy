# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 20:15:01 2021

@author: Richard Barrie
"""
import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

pole_freq= (26.56)*1e9/(2*np.pi)
z = 13e9/(2*np.pi)

b,a = sp.signal.zpk2tf([z, z],[pole_freq, pole_freq,pole_freq],-1*(pole_freq**3)/z**2)
w, mag, phase = sp.signal.bode((b,a))
plt.figure()
plt.grid()
plt.axvline(x=pole_freq*2*np.pi,color = 'grey', label = "Nyquist Frequency")
plt.legend()
plt.semilogx(w*2*np.pi, mag)
plt.title("CTLE Frequency Response Magnitude for Nyquist f = 26.56G")
plt.show()


#%%

#define network
network = rf.Network('./touchstone/DPO_4in_Meg7_THRU.s4p')
port_def = np.array([[0, 1],[2, 3]])
H,f,h,t = sdp.four_port_to_diff(network,port_def)
nyquist_f = 26.56e9
nyquist_T = 1/nyquist_f
oversampling_ratio = 64
steps_per_symbol = int(round(oversampling_ratio/2))
t_d = nyquist_T/oversampling_ratio
H, f, h, t = sdp.zero_pad(H,f,t_d)

w = f/(2*np.pi)
w, H_ctle = sp.signal.freqs(b, a, w)
H_ctle /= max(abs(H_ctle))

mag = 20*np.log10(H_ctle)

plt.figure()
plt.semilogx(1e-9*f,20*np.log10(abs(H)), label = "Channel Response")
plt.ylabel('Mag. Response [dB]')
plt.xlabel('Frequency [GHz]')

plt.semilogx(1e-9*f, mag, label = "CTLE Response")
plt.axvline(x=26.56,color = 'grey', label = "Nyquist Frequency")
plt.legend()
plt.grid()
plt.xlim([1e-1,1e3])
plt.title("Magnitude [dB] Bode Plot")

plt.figure()
plt.semilogx(1e-9*f,abs(H), label = "Channel Response")
plt.ylabel('Mag. Response [dB]')
plt.xlabel('Frequency [GHz]')
plt.title("Magnitude Bode Plot")

plt.semilogx(1e-9*f,abs(H_ctle), label = "CTLE Response")
plt.axvline(x=26.56,color = 'grey', label = "Nyquist Frequency")
plt.legend()
plt.xlim([1e-1,1e3])
plt.grid()

#%%
h_ctle, t_ctle = sdp.freq2impulse(H_ctle,f)

plt.figure()
plt.plot(t_ctle*1e12, h_ctle)
plt.title("CTLE impulse response")

plt.figure()
plt.plot(t_ctle*1e12, h_ctle)
plt.xlim([-1, 10])
plt.title("CTLE impulse response (start)")

plt.figure()
plt.plot(t_ctle*1e12, h_ctle)
plt.xlim([100000-40, 100000])
plt.title("CTLE impulse response (end)")


#%%
sig = sdp.nrz_input(32,np.array([1,0,1,0,1,0,1,0,1,0,1,0]),np.array([0,1]))
ctle_pulse = sp.signal.fftconvolve(sig,h_ctle[0:1000])
plt.figure()
plt.plot(ctle_pulse)
plt.xlim([-1, 1000])
#%%

#compute input data using PRQS10
data_in = sdp.prqs10(1)

#take first 10k bits for faster simulation
data_in = data_in[:10000]

#define voltage levels for 0 and 1 bits
voltage_levels = np.array([-3, -1, 1, 3])

#convert data_in to time domain signal
signal_in = sdp.pam4_input(steps_per_symbol, data_in, voltage_levels)

signal_output = sp.signal.fftconvolve(h, signal_in,mode = 'same')

#define signal object for this signal, crop out first bit of signal which is 0 due to channel latency
sig = sdp.Receiver(signal_output[5000:], steps_per_symbol, t[1], voltage_levels)
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "PAM-4 Eye, 112 Gbit/s")

#%%
sig.reset()

sig.signal = sp.signal.fftconvolve(sig.signal,h_ctle[0:1000], mode = 'same')

sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "PAM-4 Eye with CTLE, 112 Gbit/s")

