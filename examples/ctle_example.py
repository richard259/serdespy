"""
Example of CTLE model
"""

import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#define ctle
rx_bw = 8e8
peak_freq = 8e8

peak_mag = 12
dc_offset = 0

#calculating pole and zero frequencies of CTLE TF for desired peaking and gain
p2 = -2.0 * np.pi * rx_bw
p1 = -2.0 * np.pi * peak_freq
z = p1 / pow(10.0, peak_mag / 20.0)

#calculating coefficients of transfer function in standard form
b,a = sp.signal.zpk2tf([z],[p1, p2],1)
b *= 1/(b[-1]/a[-1])

#define channel response
#network = rf.Network('./touchstone/DPO_4in_Meg7_THRU.s4p')
network = rf.Network('../touchstone/C2C_PCB_SYSVIA_12dB_thru.s4p')

port_def = np.array([[0, 1],[2, 3]])
H,f,h,t = sdp.four_port_to_diff(network,port_def)
nyquist_f = 26.56e9
nyquist_T = 1/nyquist_f
oversampling_ratio = 64
samples_per_symbol = int(round(oversampling_ratio/2))
t_d = nyquist_T/oversampling_ratio
H, f, h, t = sdp.zero_pad(H,f,t_d)

#frequency vector in rad/s
w = f/(2*np.pi)

#calculate Frequency response of CTLE at given frequencies
w, H_ctle = sp.signal.freqs(b, a, w)

#magnitude of Frequency response of transfer function for given frequencies
mag = 20*np.log10(H_ctle)


#plot frequency response of channel and ctle
plt.figure()
plt.semilogx(1e-9*f,20*np.log10(abs(H)), label = "Channel Response")
plt.ylabel('Mag. Response [dB]')
plt.xlabel('Frequency [GHz]')


plt.semilogx(1e-9*f, mag, label = "CTLE Response")
plt.axvline(x=26.56,color = 'grey', label = "Nyquist Frequency")
plt.legend()
plt.grid()
plt.xlim([1e-1,1e3])
plt.ylim([-25,10])
plt.title("Magnitude [dB] Bode Plot")
plt.savefig('CTLE1.png')

plt.figure()
plt.semilogx(1e-9*f,abs(H), label = "Channel Response")
plt.ylabel('Mag. Response')
plt.xlabel('Frequency [GHz]')
plt.title("Magnitude Bode Plot")

plt.semilogx(1e-9*f,abs(H_ctle), label = "CTLE Response")
plt.axvline(x=26.56,color = 'grey', label = "Nyquist Frequency")
plt.legend()
plt.xlim([1e-1,1e3])
plt.grid()
plt.savefig('CTLE2.png')
#%% compute and plot impulse response of CTLE
h_ctle, t_ctle = sdp.freq2impulse(H_ctle,f)

#plt.figure()
#plt.plot(t_ctle*1e9,h_ctle)
#plt.title("Full CTLE Impulse Response")
#plt.ylabel('CTLE Impulse Response [V]')
#plt.xlabel('Time [ns]')


#take start of CTLE impulse response for faster convoulition (reflection at end of impulse response is not physically real)

h_ctle = h_ctle[:100]
t_ctle = t_ctle[:100]

#pl#t.figure()
#pl#t.plot(t_ctle*1e12,h_ctle)
#p#lt.title("Start of CTLE Impulse Response")
#plt.ylabel('CTLE Impulse Response [V]')
#plt.xlabel('Time [ps]')
#plt.xlim([-5,])


#%% measure precursor and postcursor from pulse response

n_taps_post = 10
n_taps_pre = 3
n_taps = n_taps_post+n_taps_pre+1

pulse_input = np.ones(samples_per_symbol)

ch_pulse_response = sp.signal.fftconvolve(h, pulse_input)
ch_pulse_response = ch_pulse_response[:t.size]

plt.figure()
plt.plot(t,ch_pulse_response)
plt.title("channel pulse response")
channel_coefficients =  sdp.channel_coefficients(ch_pulse_response, t, samples_per_symbol, n_taps_pre, n_taps_post) 

ch_ctle_pulse_response = sp.signal.fftconvolve(ch_pulse_response, h_ctle)
ch_ctle_pulse_response = ch_ctle_pulse_response[:t.size]

plt.figure()
plt.plot(t,ch_ctle_pulse_response)
plt.title("channel + ctle pulse response")

channel_coefficients =  sdp.channel_coefficients(ch_ctle_pulse_response, t, samples_per_symbol, n_taps_pre, n_taps_post) 

#%% solve for zero-forcing FFE tap weights
    
A = np.zeros((n_taps,n_taps))

for i in range(n_taps):
    A += np.diag(np.ones(n_taps-abs(i-n_taps_pre))*channel_coefficients[i],(n_taps_pre-i) )

C = np.zeros((n_taps,1))
C[n_taps_pre] = 1

B = np.linalg.inv(A)@C

B = B/np.sum(abs(B))

ffe_tap_weights = B.T[0]

#%%create TX waveform

data_in = sdp.prbs13(1)

#define voltage levels for 0 and 1 bits
voltage_levels = np.array([-0.5, 0.5])

#convert data_in to time domain signal
signal_in = sdp.nrz_input(samples_per_symbol, data_in, voltage_levels)

sdp.simple_eye(signal_in, samples_per_symbol*3, 1000, t[1], "nrz tx wf")

#do convolution to get differential channel response
signal_output = sp.signal.fftconvolve(signal_in, h, mode = 'same')
#signal_output = signal_output[0:h.size]

sig = sdp.Receiver(signal_output[6000:], samples_per_symbol, t[1], voltage_levels)


#%% plot eye diagrams with FFE 

#no FFE
sig.reset()
sdp.simple_eye(sig.signal, sig.samples_per_symbol*2, 1000, sig.t_step, "NRZ Eye, 53 Gbit/s")
plt.savefig('CTLE3.png')
np.save("nrz_eye",sig.signal)

sig.CTLE(b,a,f)
sdp.simple_eye(sig.signal, sig.samples_per_symbol*2, 1000, sig.t_step, "NRZ Eye, 53 Gbit/s with CTLE")
plt.savefig('CTLE4.png')
np.save("nrz_eye_ctle",sig.signal)

#with FFE and computed weights
sig.FFE(ffe_tap_weights,n_taps_pre)
sdp.simple_eye(sig.signal[32*10:], sig.samples_per_symbol*2, 1000, sig.t_step, "NRZ Eye, 53 Gbit/s with CTLE and Zero-Forcing FFE")
plt.savefig('CTLE5.png')
np.save("nrz_eye_ctle_ffe",sig.signal)
#%%create TX waveform

#compute input data using PRQS10
data_in = sdp.prqs10(1)

#take first 10k bits for faster simulation
data_in = data_in[:10000]

#define voltage levels for 0 and 1 bits
voltage_levels = np.array([-3, -1, 1, 3])

#convert data_in to time domain signal
signal_in = sdp.pam4_input(samples_per_symbol, data_in, voltage_levels)
sdp.simple_eye(signal_in, samples_per_symbol*3, 1000, t[1], "pam4 tx wf")
#do convolution to get differential channel response
signal_output = sp.signal.fftconvolve(signal_in, h, mode = 'same')

#define signal object for this signal, crop out first bit of signal which is 0 due to channel latency
sig = sdp.Receiver(signal_output[5000:], samples_per_symbol, t[1], voltage_levels)

#%%sig.reset()
sdp.simple_eye(sig.signal, sig.samples_per_symbol*2, 1000, sig.t_step, "PAM-4 Eye, 106 Gbit/s")
plt.savefig('CTLE6.png')

sig.CTLE(b,a,f)
sdp.simple_eye(sig.signal, sig.samples_per_symbol*2, 1000, sig.t_step, "PAM-4 Eye, 106 Gbit/s with CTLE")
plt.savefig('CTLE7.png')
np.save("pam4_eye_ctle",sig.signal)

#with FFE and computed weights
sig.FFE(ffe_tap_weights,n_taps_pre)
sdp.simple_eye(sig.signal, sig.samples_per_symbol*2, 1000, sig.t_step, "PAM-4 Eye, 106 Gbit/s with CTLE and Zero-Forcing FFE")
plt.savefig('CTLE8.png')