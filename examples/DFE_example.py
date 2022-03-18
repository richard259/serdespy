"""Example of channel coefficient calculation, zero-forcing FFE operation and Bit Error Rate Calculation"""

import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#define network
network = rf.Network('../touchstone/C2C_PCB_SYSVIA_12dB_thru.s4p')

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
samples_per_symbol = int(round(oversampling_ratio/2))

#Desired time-step
t_d = nyquist_T/oversampling_ratio

#compute response of zero-padded TF
H, f, h, t = sdp.zero_pad(H,f,t_d)

#%% convert pulse response to channel coefficients

n_taps_post = 8
n_taps_pre = 0
n_taps = n_taps_post+n_taps_pre+1

pulse_input = np.ones(samples_per_symbol)

pulse_response = sp.signal.fftconvolve(h, pulse_input, mode = "same")
plt.savefig('DFE1.png')
channel_coefficients =  sdp.channel_coefficients(pulse_response, t, samples_per_symbol, n_taps_pre, n_taps_post) 
#%%
#print(channel_coefficients)
tap_weights = (channel_coefficients)[1:]
#%% Create RX eye diagram with and without FFE.  

#use only 2000 symbols for faster simulation to construct eye diagram
data_in = sdp.prbs20(1)
data_in= data_in[:6000]

#create ideal transmitter waveform
signal_in = sdp.nrz_input(samples_per_symbol,data_in,np.array([-1,1]))

#convolve with channel response
signal_out = sp.signal.fftconvolve(signal_in, h)[5000:300000]

#set up Reciever Class with output signal
RX = sdp.Receiver(np.copy(signal_out),samples_per_symbol,nyquist_T, np.array([-0.5,0.5]))

#create eye diagram with no FFE
sdp.simple_eye(RX.signal, samples_per_symbol*3, 1000, t_d, "53 Gbit/s RX Eye Diagram")
plt.savefig('DFE2.png')
#Apply zero-forcing FFE
RX.nrz_DFE(tap_weights)

#Create Eye diagram with FFE
sdp.simple_eye(RX.signal, samples_per_symbol*3, 1000, t_d, "53 Gbit/s RX Eye Diagram with DFE")
plt.savefig('DFE3.png')
#%%
#use only 2000 symbols for faster simulation to construct eye diagram
data_in = sdp.prqs10(1)
data_in= data_in[:6000]

#create ideal transmitter waveform
signal_in = sdp.pam4_input(samples_per_symbol,data_in,np.array([-3,-1,1,3]))

#convolve with channel response
signal_out = sp.signal.fftconvolve(signal_in, h)[5000:300000]

#set up Reciever Class with output signal
#%%
RX = sdp.Receiver(np.copy(signal_out),samples_per_symbol,nyquist_T,np.array([-3,-1,1,3]))

#create eye diagram with no FFE
sdp.simple_eye(RX.signal, samples_per_symbol*3, 1000, t_d, "106 Gbit/s RX Eye Diagram")
plt.savefig('DFE4.png')
#Apply zero-forcing FFE
RX.pam4_DFE(tap_weights)

#Create Eye diagram with FFE
sdp.simple_eye(RX.signal, samples_per_symbol*3, 1000, t_d, "106 Gbit/s RX Eye Diagram with DFE")
plt.savefig('DFE5.png')