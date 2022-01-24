"""
Example usage of Transmitter class
"""

import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#%% set up some parameters for the transmitter waveform

voltage_levels = np.array([-3,-1,1,3])
samples_per_symbol=128
#nyquist_f = 26.56e9

#frequency is 5G 
nyquist_f = 5e9

#%% Create data from PRQS, use first 2000 symbols only for faster simulations
data = sdp.prqs10(1)
data = data[:2000]

#%% Create transmitter class
TX = sdp.Transmitter(data, voltage_levels, nyquist_f)


#%% Apply FIR
tap_weights = np.array([-0, 1, -0.1])
TX.FIR(tap_weights)

#%%Oversample the TX waveform
TX.oversample(samples_per_symbol)

#plot eye diagram
sdp.simple_eye(TX.signal_ideal, samples_per_symbol*3, 500, TX.UI/TX.samples_per_symbol, "TX Eye Diagram Ideal")

#%%Add gaussian jitter at 10% of UI

TX.gaussian_jitter(0.1)

#plot eye diagram
sdp.simple_eye(TX.signal, samples_per_symbol*3, 500, TX.UI/TX.samples_per_symbol, "TX Eye Diagram Jitter")

#%%Add bandwidth

TX.tx_bandwidth()
sdp.simple_eye(TX.signal, samples_per_symbol*3, 500, TX.UI/TX.samples_per_symbol, "TX Eye Diagram Bandwidth")

#%%create channel model impulse response

network = rf.Network('touchstone/DPO_4in_Meg7_THRU.s4p')

port_def = np.array([[0, 1],[2, 3]])

#get TF of differential network
H,f,h,t = sdp.four_port_to_diff(network,port_def)

#Period of clock at nyquist frequency
nyquist_T = 1/nyquist_f

#Desired time-step
t_d = TX.UI/samples_per_symbol

#compute response of zero-padded TF
H, f, h, t = sdp.zero_pad(H,f,t_d)


#%%compute signal at RX side, and plot without jitter

channel_output = sp.signal.fftconvolve(TX.signal_ideal,h,mode="same")

sdp.simple_eye(channel_output, samples_per_symbol*3, 500, TX.UI/samples_per_symbol, "RX Eye Diagram")


#%%compute signal at RX side, and plot with jitter

channel_output = sp.signal.fftconvolve(TX.signal,h,mode="same")

sdp.simple_eye(channel_output, samples_per_symbol*3, 500, TX.UI/samples_per_symbol, "RX Eye Diagram")


