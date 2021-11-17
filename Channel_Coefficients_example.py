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
nyquist_f = 26.56e9*1.69

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

#%% convert pulse response to channel coefficients

n_taps_post = 10
n_taps_pre = 4
n_taps = n_taps_post+n_taps_pre+1

pulse_input = np.ones(steps_per_symbol)

pulse_response = sp.signal.fftconvolve(h, pulse_input)[:h.size]

channel_coefficients =  sdp.channel_coefficients(pulse_response, t, steps_per_symbol, n_taps_pre, n_taps_post) 

#%%
A = np.zeros((n_taps,n_taps))

for i in range(n_taps):
    A += np.diag(np.ones(n_taps-abs(i-n_taps_pre))*channel_coefficients[i],(n_taps_pre-i) )

c = np.zeros((n_taps,1))
c[n_taps_pre] = 1

b = np.linalg.inv(A)@c

b = b/np.sum(abs(b))

ffe_tap_weights = b.T[0]


#%%
data_in = sdp.prbs20(1)

signal_in = np.copy(sdp.nrz_input(1,data_in,np.array([-0.5,0.5])))

signal_out = sp.signal.fftconvolve(channel_coefficients, signal_in)

RX = sdp.Receiver(np.copy(signal_out[4:-10]),1,nyquist_T, np.array([-0.5,0.5]),shift=False)

RX.FFE(ffe_tap_weights,4)

#%%

data_out = sdp.nrz_a2d(RX.signal[100:-100],1,0)

print(sdp.prbs_checker(20,data_in,data_out)[0])