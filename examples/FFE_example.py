"""Example of channel coefficient calculation, zero-forcing FFE operation and Bit Error Rate Calculation"""

import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#define network
network = rf.Network('./touchstone/DPO_4in_Meg7_THRU.s4p')

#set up port definition of network
port_def = np.array([[0, 1],[2, 3]])

#get TF of differential network
H,f,h,t = sdp.four_port_to_diff(network,port_def)

#Nyquist frequency
nyquist_f = 26.56e9*1

#Period of clock at nyquist frequency
nyquist_T = 1/nyquist_f

#desired number of samples per clock period
oversampling_ratio = 64

#timesteps per bit
samples_per_symbol = int(round(oversampling_ratio/2))

#Desired time-step
t_d = nyquist_T/oversampling_ratio

#compute response of zero-padded TF
H, f, h, t = sdp.zero_pad(H,f,t_d)

#%% convert pulse response to channel coefficients

n_taps_post = 8
n_taps_pre = 4
n_taps = n_taps_post+n_taps_pre+1

pulse_input = np.ones(samples_per_symbol)

pulse_response = sp.signal.fftconvolve(h, pulse_input, mode = "same")

channel_coefficients =  sdp.channel_coefficients(pulse_response, t, samples_per_symbol, n_taps_pre, n_taps_post) 

#%%zero forcing algorithm

A = np.zeros((n_taps,n_taps))

for i in range(n_taps):
    A += np.diag(np.ones(n_taps-abs(i-n_taps_pre))*channel_coefficients[i],(n_taps_pre-i) )

c = np.zeros((n_taps,1))
c[n_taps_pre] = 1

b = np.linalg.inv(A)@c

#normalize tap weights
b = b/np.sum(abs(b))

ffe_tap_weights = b.T[0]


#%% Create RX eye diagram with and without FFE.  

#use only 2000 symbols for faster simulation to construct eye diagram
data_in = sdp.prbs20(1)
data_in= data_in[:2000]

#create ideal transmitter waveform
signal_in = sdp.nrz_input(samples_per_symbol,data_in,np.array([-0.5,0.5]))

#convolve with channel response
signal_out = sp.signal.fftconvolve(signal_in, h)[5000:65000]

#set up Reciever Class with output signal
RX = sdp.Receiver(np.copy(signal_out),samples_per_symbol,nyquist_T, np.array([-0.5,0.5]))

#create eye diagram with no FFE
sdp.simple_eye(RX.signal, samples_per_symbol*3, 500, t_d, "RX Eye Diagram no FFE")

#Apply zero-forcing FFE
RX.FFE(ffe_tap_weights, n_taps_pre)

#Create Eye diagram with FFE
sdp.simple_eye(RX.signal, samples_per_symbol*3, 500, t_d, "RX Eye Diagram FFE")

#%% Run Baud-Rate simulation on entire PRBS20 pattern to get bit error rate estimates with and without FFE

#create entire 1M bit long prbs sequence
data_in = sdp.prbs20(1)

#create BR sampled TX waveform
signal_in = np.copy(sdp.nrz_input(1,data_in,np.array([-0.5,0.5])))

#convolve with channel coefficients
signal_out = sp.signal.fftconvolve(signal_in, channel_coefficients, mode = 'same')

#Set up Receiver class with BR sampled RX waveform
RX = sdp.Receiver(np.copy(signal_out[4:-10]),1,nyquist_T, np.array([-0.5,0.5]),shift=False)

#slice output waveform to recover data
data_out = sdp.nrz_a2d(RX.signal[1000:-1000],1,0)

#calculate BER with no FFE
print("NO FFE:", sdp.prbs_checker(20,data_in,data_out)[0], "Errors in", data_out.size, "Bits Simulated")

print("BER = ", sdp.prbs_checker(20,data_in,data_out)[0]/data_out.size)

#apply FFE
RX.FFE(ffe_tap_weights, n_taps_pre)

#calculate BER with FFE
data_out = sdp.nrz_a2d(RX.signal[1000:-1000],1,0)

print("FFE:", sdp.prbs_checker(20,data_in,data_out)[0], "Errors in", data_out.size, "Bits Simulated")

print("BER = ", sdp.prbs_checker(20,data_in,data_out)[0]/data_out.size)
