"""Example of FFE operation"""

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
nyquist_f = 26.56e9

#Period of clock at nyquist frequency
nyquist_T = 1/nyquist_f

#desired number of samples per clock period
n = 64

#timesteps per bit
steps_per_symbol = int(round(n/2))

#Desired time-step
t_d = nyquist_T/n

#compute response of zero-padded TF
H, f, h, t = sdp.zero_pad(H,f,t_d)

#%%create TX waveform

#compute input data using PRBS13
data_in = sdp.prbs13(1)

#define voltage levels for 0 and 1 bits
voltage_levels = np.array([-0.5, 0.5])

#convert data_in to time domain signal
signal_in = sdp.nrz_input(steps_per_symbol, data_in, voltage_levels)

#%%compute channel response to signal_in

#TODO: figure out convolution with different length vectors

h_zero_pad = np.hstack((h, np.zeros(signal_in.size-h.size)))

#do convolution to get differential channel response
signal_output = sp.signal.fftconvolve(h_zero_pad, signal_in)
signal_output = signal_output[0:h_zero_pad.size]

#define signal object for this signal, crop out first bit of signal which is 0 due to channel latency
sig = sdp.Receiver(signal_output[5000:], steps_per_symbol, t[1], voltage_levels)


#%% measure precursor and postcursor from pulse response
half_symbol = int(round(n/4))

#create pulse waveform
pulse_input = np.ones(steps_per_symbol)
pulse_response = np.convolve(h, pulse_input,mode='same')

#find peak of pulse response
max_idx = np.where(pulse_response == np.amax(pulse_response))[0][0]

#number of DFE taps
n_dfe_taps = 8

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
print(dfe_tap_weights)
plt.figure()
plt.plot(np.linspace(int((pc[0])-150),int(pc[-1]),int(pc[-1]-pc[0]+151)),pulse_response[int(pc[0])-150:int(pc[-1]+1)],label = 'Pulse Response')
plt.plot(pc, dfe_tap_weights, 'o',label = 'Tap Weights')
plt.xlabel("Time [s]")
plt.ylabel("impulse response [V]")
plt.title("Tap Weight Estimation From Pulse Response")
plt.legend()
for xc in xcoords:
    plt.axvline(x=xc,color = 'grey')

#%%
n_ffe_taps_post = 1
n_ffe_taps_pre = 1
n_ffe_taps = n_ffe_taps_post+n_ffe_taps_pre+1

pulse_input = np.ones(steps_per_symbol)

pulse_response = np.convolve(h, pulse_input,mode='same')

channel_coefficients =  sdp.channel_coefficients(pulse_response, t, steps_per_symbol, n_ffe_taps_pre, n_ffe_taps_post) 

#%% solve for zero-forcing FFE tap weights
    
A = np.zeros((n_dfe_taps+1+n_ffe_taps_pre,n_ffe_taps))

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if i == j:
            A[i][j] = channel_coefficients[1]
        if i-j ==1:
            A[i][j] = channel_coefficients[2]
        if i-j == -1:
            A[i][j] = channel_coefficients[0]

c = np.zeros((n_dfe_taps+1+n_ffe_taps_pre,1))
c[n_ffe_taps_pre] = 1
c[n_ffe_taps_pre+1:] = dfe_tap_weights.reshape(8,1)


b = np.linalg.pinv(A)@c

b = b/np.sum(abs(b))

ffe_tap_weights = b.T[0]


#%%create TX waveform

#compute input data using PRBS13
data_in = sdp.prbs13(1)

#define voltage levels for 0 and 1 bits
voltage_levels = np.array([-0.5, 0.5])

#convert data_in to time domain signal
signal_in = sdp.nrz_input(steps_per_symbol, data_in, voltage_levels)
#%%compute channel response to signal_in

h_zero_pad = np.hstack((h, np.zeros(signal_in.size-h.size)))

#do convolution to get differential channel response
signal_output = sp.signal.fftconvolve(h_zero_pad, signal_in)
signal_output = signal_output[0:h_zero_pad.size]

#define signal object for this signal, crop out first bit of signal which is 0 due to channel latency

sig = sdp.Receiver(signal_output[5000:], steps_per_symbol, t[1], voltage_levels)
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "Eye Diagram")
#%%
#apply FFE only and show eye diagram
sig.signal = sig.signal_org
sig.DFE(dfe_tap_weights,0)
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 3000, sig.t_step, "Eye Diagram with DFE")

#apply DFE and show eye diagram
sig.signal = sig.signal_org
sig.DFE(dfe_tap_weights,0)
sig.FFE(ffe_tap_weights,n_ffe_taps_pre)
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 3000, sig.t_step, "Eye Diagram with FFE and DFE")

#%%measure BER for different levels of noise

noise_levels = np.linspace(0.03,0.05,10)
BER = np.zeros(10)

for i in range(noise_levels.size):
    sig.noise(noise_levels[i])
    sig.FFE(ffe_tap_weights,1)
    sig.DFE(dfe_tap_weights,0)
    data = sdp.nrz_a2d(sig.signal, steps_per_symbol, 0)
    errors = sdp.prbs_checker(13,data_in, data)
    BER[i] = errors[0]/data.size

#plot BER vs noise level
plt.figure()
plt.plot(noise_levels, BER, 'o')
plt.xlabel("noise stdev [V]")
plt.ylabel("BER")
plt.title("BER vs. RX noise stdev with DFE")

