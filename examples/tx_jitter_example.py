import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#%%

voltage_levels = np.array([-3,-1,1,3])
samples_per_symbol=128
#nyquist_f = 26.56e9
nyquist_f = 5e9

#%%
data = sdp.prqs10(1)

#%%
data = data[:2000]

TX = sdp.Transmitter(data, voltage_levels, nyquist_f)

tap_weights = np.array([-0, 1, -0.1])


#TX.FIR(tap_weights)

TX.oversample(samples_per_symbol)


#%%

sdp.simple_eye(TX.signal_ideal, samples_per_symbol*3, 500, TX.UI/TX.samples_per_symbol, "TX Eye Diagram Ideal")

#%%

TX.gaussian_jitter()

sdp.simple_eye(TX.signal, samples_per_symbol*3, 500, TX.UI/TX.samples_per_symbol, "TX Eye Diagram Jitter Ideal")


#%%

#set up port definition of network

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


#%%%

channel_coefficients = sdp.channel_coefficients(sp.signal.fftconvolve(h,np.ones(samples_per_symbol),mode="same"), t, samples_per_symbol, 2, 3)

channel_coefficients = channel_coefficients*(1/channel_coefficients[2])
#%%


channel_output = sp.signal.fftconvolve(TX.signal_ideal,h,mode="same")

sdp.simple_eye(channel_output, samples_per_symbol*3, 500, TX.UI/samples_per_symbol, "RX Eye Diagram")


#%%
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



