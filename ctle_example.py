import pylab
import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



#%%


rx_bw = 10e8
peak_freq = 10e8
peak_mag = 12
dc_offset = 0

p2 = -2.0 * np.pi * rx_bw
p1 = -2.0 * np.pi * peak_freq
z = p1 / pow(10.0, peak_mag / 20.0)

if p2 != p1:
    r1 = (z - p1) / (p2 - p1)
    r2 = 1 - r1
else:
    r1 = -1.0
    r2 = z - p1
    
b, a = sp.signal.invres([r1, r2], [p1, p2], [])


f = np.linspace(0,1e12,100000)
w = f/(2*np.pi)

w, H_ctle = sp.signal.freqs(b, a, w)

H_ctle /= max(abs(H_ctle))

mag = 20*np.log10(H_ctle)

plt.figure()

plt.semilogx(f, mag)    # Bode magnitude plot
plt.grid()

h_ctle, t_ctle = sdp.freq2impulse(H_ctle,f)

plt.figure()
plt.plot(t_ctle*1e12,h_ctle)

plt.xlabel("time [ps]")
plt.ylabel("CTLE impulse response [V]")

plt.xlim([0, 10])

#%%
#define network
network = rf.Network('./DPO_4in_Meg7_THRU.s4p')

#set up port definition of network
port_def = np.array([[0, 1],[2, 3]])

#get TF of differential network
H,f,h,t = sdp.four_port_to_diff(network,port_def)

#Nyquist frequency
nyquist_f = 26.56e9

#Period of clock at nyquist frequency
nyquist_T = 1/nyquist_f

#desired number of samples per clock period
oversampling_ratio = 64

#timesteps per bit
steps_per_symbol = int(round(oversampling_ratio/2))

#Desired time-step
t_d = nyquist_T/oversampling_ratio

#compute response of zero-padded TF
H, f, h, t = sdp.zero_pad(H,f,t_d)

pylab.semilogx(1e-9*f,20*np.log10(abs(H)))
plt.ylabel('Channel Mag. Response [dB]')
plt.xlabel('Frequency [GHz]')

w = f/(2*np.pi)

w, H_ctle = sp.signal.freqs(b, a, w)

H_ctle /= max(abs(H_ctle))

mag = 20*np.log10(H_ctle)

plt.figure()

plt.semilogx(f, mag)    # Bode magnitude plot
plt.grid()

h_ctle, t_ctle = sdp.freq2impulse(H_ctle,f)

plt.figure()
plt.plot(t_ctle*1e12,h_ctle)

plt.xlabel("time [ps]")
plt.ylabel("CTLE impulse response [V]")

plt.xlim([0, 10])

#%%
#%%create TX waveform

data_in = sdp.prbs13(1)

#define voltage levels for 0 and 1 bits
voltage_levels = np.array([-0.5, 0.5])

#convert data_in to time domain signal
signal_in = sdp.nrz_input(steps_per_symbol, data_in, voltage_levels)

#compute channel response to signal_in

#do convolution to get differential channel response
signal_output = sp.signal.fftconvolve(h, signal_in)
signal_output = signal_output[0:h.size]

#define Reciever object for this signal at the RX, crop out first few ns of signal

sig = sdp.Receiver(signal_output[5000:], steps_per_symbol, t[1], voltage_levels)

sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "NRZ Eye")

sig.signal = sp.signal.fftconvolve(h_ctle, sig.signal)

sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "NRZ Eye with CTLE")