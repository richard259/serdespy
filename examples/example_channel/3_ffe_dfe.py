"""
This file shows example of FFE and DFE. Equalization of the signal with CTLE generated in 2_ctle.py
"""

#import useful packages
import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp

#load in data, and set up paramaters
h = np.load("./data/h_thru.npy")
h_ctle = np.load("./data/h_ctle.npy")
samples_per_symbol = 64
nyquist_f = 26.56e9

#pulse response of channel is convolution of impulse response and 1 UI of ones
pulse_response = sp.signal.fftconvolve(h, np.ones(samples_per_symbol), mode = "full")

#plot channel coefficients for pulse response
sdp.channel_coefficients(pulse_response, np.linspace(1,pulse_response.size,pulse_response.size), samples_per_symbol, 3, 20, title = "Pulse Response")

#pulse response with CTLE
pulse_response_ctle = sp.signal.fftconvolve(pulse_response, h_ctle, mode = "full")

#plot channel coefficients. observer lower post-cursor ISI
sdp.channel_coefficients(pulse_response_ctle, np.linspace(1,pulse_response_ctle.size,pulse_response_ctle.size), samples_per_symbol, 3, 20, title = "Pulse Response with CTLE")

#%% pick 1 tap TX FIR FILTER 

#arbitrarily take -0.05 tap weight to reduce precursor ISI
tx_fir_tap_weights = np.array([-0.05, 1])

#oversample
pulse_response_fir = sp.signal.fftconvolve(h, np.repeat(tx_fir_tap_weights,samples_per_symbol), mode = "full")

#convolution with pulse response
pulse_response_fir_ctle = sp.signal.fftconvolve(pulse_response_fir, h_ctle, mode = "full")

#plot channel coefficients
channel_coefficients = sdp.channel_coefficients(pulse_response_fir_ctle, np.linspace(1,pulse_response_fir_ctle.size,pulse_response_fir_ctle.size), samples_per_symbol, 3, 20, title = "Pulse Response with FIR and CTLE")[:4]


#%% Pick 3 precursor tap FFE

n_taps_pre = 3

#calculate RX FFE coeffiecients to force precursor ISI to 0
ffe_tap_weights = sdp.forcing_ffe(n_taps_pre, channel_coefficients)

#oversample ffe tap weights so we can perform convolution on 64X oversampled sign
rx_ffe_conv = np.zeros(64*ffe_tap_weights.size)

for i in range(ffe_tap_weights.size):
    rx_ffe_conv[i*64] = ffe_tap_weights[i]

#apply FFE to pulse response
pulse_response_fir_ctle_ffe = sp.signal.fftconvolve(pulse_response_fir_ctle, rx_ffe_conv, mode = "full")

#plot channel coefficients. observe lower pre-cursor ISI
channel_coefficients = sdp.channel_coefficients(pulse_response_fir_ctle_ffe, np.linspace(1,pulse_response_fir_ctle_ffe.size,pulse_response_fir_ctle_ffe.size), samples_per_symbol, 3, 20)

#amplitude of main cursor
main_cursor = channel_coefficients[3]

#zero-forcing DFE weights
dfe_tap_weights = channel_coefficients[4:]

#%% do time-domain simulation and plot eye diagrams

voltage_levels = np.array([-3,-1,1,3])

data = sdp.prqs10(1)

TX = sdp.Transmitter(data[:10000], voltage_levels, nyquist_f)

#apply FIR filter to transmitter waveform
TX.FIR(tx_fir_tap_weights)

#oversample data to 64 samples/symbols
TX.oversample(samples_per_symbol)

#eye diagram of transmitter waveform
sdp.simple_eye(TX.signal_ideal[64*3:], samples_per_symbol*3, 500, TX.UI/TX.samples_per_symbol, "TX Ideal Eye Diagram with FIR filter")

#signal at output of channel
signal_out = sp.signal.fftconvolve(TX.signal_ideal, h, mode = "same")[:64*500*12]

#signal at output of CTLE
signal_out_ctle = sp.signal.fftconvolve(signal_out, h_ctle, mode = "same")

#plot eye diagram
sdp.simple_eye(signal_out_ctle, samples_per_symbol*3, 1000, TX.UI/TX.samples_per_symbol, "Eye Diagram with CTLE")

#%%
RX = sdp.Receiver(signal_out_ctle, samples_per_symbol, nyquist_f, voltage_levels, shift = True, main_cursor = main_cursor)

#sdp.simple_eye(RX.signal, samples_per_symbol*3, 800, TX.UI/TX.samples_per_symbol, "Eye Diagram with CTLE")

#signal after FFE
RX.FFE(ffe_tap_weights, n_taps_pre)

#sdp.simple_eye(RX.signal, samples_per_symbol*3, 800, TX.UI/TX.samples_per_symbol, "Eye Diagram with CTLE and FFE")
#signal after DFE
RX.pam4_DFE(dfe_tap_weights)

#plot eye diagram
sdp.simple_eye(RX.signal[64*300:], samples_per_symbol*3, 1000, TX.UI/TX.samples_per_symbol, f"Eye Diagram with CTLE, FFE, and DFE")

#%%save the tap weights chosen
np.save("./data/rx_ffe_tap_weights.npy",ffe_tap_weights)
np.save("./data/rx_dfe_tap_weights.npy",dfe_tap_weights)
np.save("./data/tx_fir_tap_weights.npy",tx_fir_tap_weights)
