"""Example of PAM-4 operation with FFE"""

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
n = 64

#timesteps per bit
steps_per_symbol = int(round(n/2))

#Desired time-step
t_d = nyquist_T/n

#compute response of zero-padded TF
H, f, h, t = sdp.zero_pad(H,f,t_d)

#%%create TX waveform

#compute input data using PRBS13
data_in = sdp.prqs10(1)

#define voltage levels for 0 and 1 bits
voltage_levels = np.array([-0.75, -0.25, 0.25, 0.75])

#convert data_in to time domain signal
signal_in = sdp.pam4_input(steps_per_symbol, data_in, voltage_levels)
signal_in = signal_in[:4194300]

#%%compute channel response to signal_in

h_zero_pad = np.hstack((h, np.zeros(signal_in.size-h.size)))

#do convolution to get differential channel response
signal_output = sp.signal.fftconvolve(h_zero_pad, signal_in)
signal_output = signal_output[0:h_zero_pad.size]

#define signal object for this signal, crop out first bit of signal which is 0 due to channel latency
sig = sdp.Receiver(signal_output[5000:], steps_per_symbol, t[1], voltage_levels)


#%% meauure precursor and postcursor from pulse response

n_taps_post = 3
n_taps_pre = 1
n_taps = n_taps_post+n_taps_pre+1

pulse_input = np.ones(steps_per_symbol)

pulse_response = np.convolve(h, pulse_input,mode='same')

channel_coefficients =  sdp.channel_coefficients(pulse_response, t, steps_per_symbol, n_taps_pre, n_taps_post) 

#%% solve for zero-forcing FFE tap weights
    
A = np.zeros((n_taps,n_taps))

for i in range(n_taps):
    A += np.diag(np.ones(n_taps-abs(i-n_taps_pre))*channel_coefficients[i],(n_taps_pre-i) )

c = np.zeros((n_taps,1))
c[n_taps_pre] = 1

b = np.linalg.inv(A)@c

b = b/np.sum(abs(b))

ffe_tap_weights = b.T[0]
#%% plot eye diagrams with FFE 

#no FFE
sig.reset()
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "Eye Diagram")

#with FFE and computed weights
sig.reset()
sig.FFE(ffe_tap_weights,1)
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "Eye Diagram with FFE")

#END
#%% eye diagrams with FFE and noise

sig_pulse = sdp.Receiver(pulse_response, steps_per_symbol, t[1], voltage_levels, shift = False)
sig_pulse.FFE(ffe_tap_weights,1)

#%% get dfe tap weights

half_symbol = int(round(n/4))

#find peak of pulse response
max_idx = np.where(sig_pulse.signal == np.amax(sig_pulse.signal))[0][0]

#number of DFE taps
n_taps = 8

dfe_tap_weights = np.zeros(n_taps)
pc = np.zeros(n_taps)
xcoords = []

#Estimate tap weights based on average value of each postcursor
for i in range(n_taps):
    xcoords = xcoords + [max_idx-half_symbol+i*steps_per_symbol]
    dfe_tap_weights[i] = np.average(sig_pulse.signal[max_idx+half_symbol+(i)*steps_per_symbol:max_idx+half_symbol+(i+1)*steps_per_symbol])
    pc[i] = max_idx +(i+1)*steps_per_symbol
xcoords = xcoords + [max_idx+half_symbol+i*steps_per_symbol]


#plot pulse response and tap weights
#print(tap_weights)
plt.figure()
plt.plot(np.linspace(int((pc[0])-150),int(pc[-1]),int(pc[-1]-pc[0]+151)),sig_pulse.signal[int(pc[0])-150:int(pc[-1]+1)],label = 'Pulse Response w FFE')
plt.plot(np.linspace(int((pc[0])-150),int(pc[-1]),int(pc[-1]-pc[0]+151)),pulse_response[int(pc[0])-150:int(pc[-1]+1)],label = 'Pulse Response')
plt.plot(pc, dfe_tap_weights, 'o',label = 'Tap Weights')
plt.xlabel("Time [s]")
plt.ylabel("impulse response [V]")
plt.title("Tap Weight Estimation From Pulse Response")
plt.legend()
for xc in xcoords:
    plt.axvline(x=xc,color = 'grey')
    #%%
    

#no FFE
sig.reset()
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "Eye Diagram")

#with FFE and computed weights
sig.reset()
sig.FFE(ffe_tap_weights,1)
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "Eye Diagram with FFE")

sig.reset()
sig.FFE(ffe_tap_weights,1)
sig.pam4_DFE(dfe_tap_weights)
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "Eye Diagram with FFE and DFE")