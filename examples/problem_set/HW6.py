import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp

#load data from homework 4
h = np.load("./data/h.npy")
t = np.load("./data/t.npy")

#generate binary data
data = sdp.prqs10(1)[:10000]

data_rate = 100e9
samples_per_symbol = 40

#generate Baud-Rate sampled signal from data
signal_BR = sdp.pam4_input_BR(data)

#oversampled signal
signal_ideal = np.repeat(signal_BR, samples_per_symbol)

sdp.simple_eye(signal_ideal, samples_per_symbol*3, 500, 500*1e-15, "{}Gbps Ideal 4-PAM Signal".format(data_rate/1e9))

#%%

#linewidth for seeing eye diagrams
lw = 0.05

#TX signal with jitter
signal_jitter = sdp.gaussian_jitter(signal_ideal, 20e-12, 10000, samples_per_symbol, stdev=1000e-15)

#eye diagram of TX with jitter
sdp.simple_eye(signal_jitter, samples_per_symbol*3, 500, 500*1e-15, "{}Gbps 4-PAM Signal with jitter".format(data_rate/1e9),linewidth=lw)

#signal at receiver with no jitter
signal_out_ideal = sp.signal.convolve(h,signal_ideal)
sdp.simple_eye(signal_out_ideal[100*samples_per_symbol:], samples_per_symbol, 5000, 500*1e-15, "rx eye diagram no jitter",linewidth=lw)

#signal at reciever with tx jitter
signal_out_jitter_tx = sp.signal.convolve(h,signal_jitter)
sdp.simple_eye(signal_out_jitter_tx[100*samples_per_symbol:], samples_per_symbol, 5000,  500*1e-15, "rx eye diagram with tx jitter",linewidth=lw)

#signal at receiver with rx jitter
sdp.rx_jitter_eye(signal_out_ideal[100*samples_per_symbol:],samples_per_symbol,5000,5000,500*1e-15,"rx eye diagram with rx_jitter",stdev=1000e-15,linewidth=lw)
