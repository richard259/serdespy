import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#%%

#define network
network = rf.Network('touchstone/DPO_4in_Meg7_THRU.s4p')

voltage_levels = np.array([-1,1])
samples_per_symbol=128
#nyquist_f = 26.56e9
nyquist_f = 10e9

#%%
#set up port definition of network
port_def = np.array([[0, 1],[2, 3]])

#get TF of differential network
H,f,h,t = sdp.four_port_to_diff(network,port_def)

#Period of clock at nyquist frequency
nyquist_T = 1/nyquist_f

#Desired time-step
t_d = TX.UI/samples_per_symbol

#compute response of zero-padded TF
H, f, h, t = sdp.zero_pad(H,f,t_d)

#%%
data = sdp.prbs13(1)
TX = sdp.Transmitter(data, samples_per_symbol, nyquist_f, voltage_levels)
sdp.simple_eye(TX.signal_ideal, samples_per_symbol*3, 500, TX.UI/TX.samples_per_symbol, "TX Eye Diagram Ideal")

#jitter at 10# of UI
TX.gaussian_jitter(0.1)
sdp.simple_eye(TX.signal, samples_per_symbol*3, 500, TX.UI/TX.samples_per_symbol, "TX Eye Diagram Jitter")

#%%
channel_output_ideal = sp.signal.fftconvolve(TX.signal_ideal,h,mode="same")
sdp.simple_eye(channel_output_ideal, samples_per_symbol*3, 1000, TX.UI/samples_per_symbol, "RX Eye Diagram Ideal")

#%%
channel_output_ideal = sp.signal.fftconvolve(TX.signal,h,mode="same")
sdp.simple_eye(channel_output_ideal, samples_per_symbol*3, 1000, TX.UI/samples_per_symbol, "RX Eye Diagram Jitter")

#%%
TX.gaussian_jitter()



