"""Example of PAM-4 operation with DFE"""

import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#define network
network = rf.Network('./DPO_4in_Meg7_THRU.s4p')

#set up port definition of network
port_def = np.array([[0, 1],[2, 3]])

#get TF of differential network
H,f,h,t = sdp.four_port_to_diff(network,port_def)

#Nyquist frequency
nyquist_f = 26.56e9/2

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

#%%create TX waveform

#compute input data using PRQS10
data_in = sdp.prqs10(1)

#take first 10k bits for faster simulation
data_in = data_in[:10000]

#define voltage levels for 0 and 1 bits
voltage_levels = np.array([-0.75, -0.25, 0.25, 0.75])

#convert data_in to time domain signal
signal_in = sdp.pam4_input(steps_per_symbol, data_in, voltage_levels)

#%%compute channel response to signal_in

h_zero_pad = np.hstack((h, np.zeros(signal_in.size-h.size)))

#do convolution to get differential channel response
signal_output = sp.signal.fftconvolve(h_zero_pad, signal_in)
signal_output = signal_output[0:h_zero_pad.size]

#define signal object for this signal, crop out first bit of signal which is 0 due to channel latency
sig = sdp.Receiver(signal_output[5000:], steps_per_symbol, t[1], voltage_levels)


#%% get dfe tap weights

half_symbol = int(round(oversampling_ratio/4))

#create pulse waveform
pulse_input = np.hstack((np.ones(steps_per_symbol),np.zeros(t.size-steps_per_symbol)))

#compute pulse response
pulse_response = sp.signal.fftconvolve(h, pulse_input)
pulse_response = pulse_response[0:t.size]

#find peak of pulse response
max_idx = np.where(pulse_response == np.amax(pulse_response))[0][0]

#number of DFE taps
n_taps = 4

dfe_tap_weights = np.zeros(n_taps)
pc = np.zeros(n_taps)
xcoords = []

#Estimate tap weights based on average value of each postcursor
for i in range(n_taps):
    xcoords = xcoords + [max_idx-half_symbol+i*steps_per_symbol]
    dfe_tap_weights[i] = np.average(pulse_response[max_idx+half_symbol+(i)*steps_per_symbol:max_idx+half_symbol+(i+1)*steps_per_symbol])
    pc[i] = max_idx +(i+1)*steps_per_symbol
xcoords = xcoords + [max_idx+half_symbol+i*steps_per_symbol]


#plot pulse response and tap weights
#print(tap_weights)
plt.figure()
plt.plot(np.linspace(int((pc[0])-150),int(pc[-1]),int(pc[-1]-pc[0]+151)),pulse_response[int(pc[0])-150:int(pc[-1]+1)],label = 'Pulse Response')
plt.plot(pc, dfe_tap_weights, 'o',label = 'Tap Weights')
plt.xlabel("Time [s]")
plt.ylabel("impulse response [V]")
plt.title("Tap Weight Estimation From Pulse Response")
plt.legend()
for xc in xcoords:
    plt.axvline(x=xc,color = 'grey')


#%% plot eye diagrams with FFE 

#no FFE
sig.reset()
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "Eye Diagram")

#with FFE and computed weights
sig.reset()
sig.pam4_DFE(dfe_tap_weights)
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "PAM-4 Eye, 53 Gbit/s with 4-Tap DFE")

#%%%%

