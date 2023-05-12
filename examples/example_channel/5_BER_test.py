"""
This file shows example of simulation to test BER of link with equalization modelled in previous files

This file also includes gaussian jitter, and a bandwidth-limiting filter applied to the transmitter waveform
AWGN is added to the signal after the channel + CTLE
"""

import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp

#load data from previous files
h = np.load("./data/h_thru.npy")
h_ctle = np.load("./data/h_ctle.npy")
ffe_tap_weights = np.load("./data/rx_ffe_tap_weights.npy")
dfe_tap_weights = np.load("./data/rx_dfe_tap_weights.npy")
tx_fir_tap_weights = np.load("./data/tx_fir_tap_weights.npy")

#set up paramaters
nyquist_f = 26.56e9
samples_per_symbol = 64
voltage_levels = np.array([-3,-1,1,3])

#pseudo-random data for simulation
data = sdp.prqs10(1)

#set up transmitter waveform
TX = sdp.Transmitter(data, voltage_levels, nyquist_f)

#apply FIR filter
TX.FIR(tx_fir_tap_weights)

#oversample transmitter waveform
TX.oversample(samples_per_symbol)

#add 0-mean jitter with standard deviation that is 2.5 % of UI time
TX.gaussian_jitter(stdev_div_UI=0.025)

#low-pass filter with cutoff frequency 120G
TX.tx_bandwidth(freq_bw=120e9)

#plot eye diagram of bandwidth-limitied transmitter waveform
sdp.simple_eye(TX.signal, samples_per_symbol*3, 1000, TX.UI/TX.samples_per_symbol, "TX Bandwidth-Limited Eye Diagram (-3dB frequency at 100GHz)")
#%% signal after channel
signal_out = sp.signal.fftconvolve(TX.signal, h, mode = "same")

#optional: add xtalk interference
#xtalk = np.load("./data/xt_response.npy")
#signal_out = signal_out[:xtalk.size]+xtalk

signal_out_ctle = sp.signal.fftconvolve(signal_out, h_ctle, mode = "same")
#%%plot eye diagram with ctle
sdp.simple_eye(signal_out_ctle, samples_per_symbol*3, 1000, TX.UI/TX.samples_per_symbol, "Eye Diagram with CTLE")

main_cursor = np.max(signal_out_ctle)/np.max(voltage_levels)

RX = sdp.Receiver(signal_out_ctle, samples_per_symbol, nyquist_f, voltage_levels, shift = True, main_cursor = 0.5)

#slice signal at centre of eye
RX.slice_signal()

#add noise with standard deviation 0.1
RX.signal_BR =  RX.signal_BR + np.random.normal(scale=0.05, size = RX.signal_BR.size)

#plot baud-rate eye diagram with ctle and noise
sdp.simple_eye(RX.signal_BR, 3, 1000, TX.UI, "Eye Diagram with CTLE and noise BR")

#apply feed-forward equalization to baud-rate-sampled signal
RX.FFE_BR(ffe_tap_weights, 3)

#sdp.simple_eye(RX.signal_BR, 3, 800, TX.UI/TX.samples_per_symbol, "Eye Diagram after FFE")

# apply decision-feedback equalization 
RX.pam4_DFE_BR(dfe_tap_weights)

#plot baud-rate eye diagram after DFE
sdp.simple_eye(RX.signal_BR, 3, 800, TX.UI/TX.samples_per_symbol, "Eye Diagram after DFE")

#%%check the number of errors and calculate BER
err = sdp.prqs_checker(10, data, RX.symbols_out[10:-10])

print("Bits Transmitted =", RX.symbols_out[10:-10].size*2, 'Bit Errors =', err[0])

print("Bit Error Ratio = ", err[0]/(RX.symbols_out[10:-10].size*2))