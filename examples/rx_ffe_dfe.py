"""
This file shows example of RX FFE and DFE
"""

import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp

nyquist_f = 26.56e9
h = np.load("./data/h_thru.npy")
h_ctle = np.load("./data/h_ctle.npy")
samples_per_symbol = 64


pulse_response = sp.signal.fftconvolve(h, np.ones(samples_per_symbol), mode = "full")
sdp.channel_coefficients(pulse_response, np.linspace(1,pulse_response.size,pulse_response.size), samples_per_symbol, 3, 20, title = "Pulse Response")

pulse_response_ctle = sp.signal.fftconvolve(pulse_response, h_ctle, mode = "full")
sdp.channel_coefficients(pulse_response_ctle, np.linspace(1,pulse_response_ctle.size,pulse_response_ctle.size), samples_per_symbol, 3, 20, title = "Pulse Response with CTLE")

# pick 1 tap TX FIR coefficient to reduce precursor ISI

tx_fir_tap_weights = np.array([-0.1, 1])
pulse_response_fir = sp.signal.fftconvolve(h, np.repeat(tx_fir_tap_weights,samples_per_symbol), mode = "full")
pulse_response_fir_ctle = sp.signal.fftconvolve(pulse_response_fir, h_ctle, mode = "full")
channel_coefficients = sdp.channel_coefficients(pulse_response_fir_ctle, np.linspace(1,pulse_response_fir_ctle.size,pulse_response_fir_ctle.size), samples_per_symbol, 3, 20, title = "Pulse Response with FIR and CTLE")[:4]

#RX FFE to force precursor ISI to 0
n_taps_pre = 3

ffe_tap_weights = sdp.forcing_ffe(n_taps_pre, channel_coefficients)

#%%
rx_ffe_conv = np.zeros(64*ffe_tap_weights.size)

for i in range(ffe_tap_weights.size):
    rx_ffe_conv[i*64] = ffe_tap_weights[i]
    
pulse_response_fir_ctle_ffe = sp.signal.fftconvolve(pulse_response_fir_ctle, rx_ffe_conv, mode = "full")

channel_coefficients = sdp.channel_coefficients(pulse_response_fir_ctle_ffe, np.linspace(1,pulse_response_fir_ctle_ffe.size,pulse_response_fir_ctle_ffe.size), samples_per_symbol, 3, 8)

main_cursor = channel_coefficients[3]
dfe_tap_weights = channel_coefficients[4:]
#%%
voltage_levels = np.array([-3,-1,1,3])

data = sdp.prqs10(1)

TX = sdp.Transmitter(data[:10000], voltage_levels, nyquist_f)

TX.FIR(tx_fir_tap_weights)

TX.oversample(samples_per_symbol)

sdp.simple_eye(TX.signal_ideal[64*3:], samples_per_symbol*3, 500, TX.UI/TX.samples_per_symbol, "TX Ideal Eye Diagram with FFE")

#%%
signal_out = sp.signal.fftconvolve(TX.signal_ideal, h, mode = "same")[:64*500*12]

signal_out_ctle = sp.signal.fftconvolve(signal_out, h_ctle, mode = "same")

sdp.simple_eye(signal_out_ctle, samples_per_symbol*3, 1000, TX.UI/TX.samples_per_symbol, "Eye Diagram with CTLE")

#%%
RX = sdp.Receiver(signal_out_ctle, samples_per_symbol, nyquist_f, voltage_levels, shift = True, main_cursor = main_cursor)

#sdp.simple_eye(RX.signal, samples_per_symbol*3, 800, TX.UI/TX.samples_per_symbol, "Eye Diagram with CTLE")

RX.FFE(ffe_tap_weights, n_taps_pre)

sdp.simple_eye(RX.signal, samples_per_symbol*3, 800, TX.UI/TX.samples_per_symbol, "Eye Diagram with CTLE and FFE")

RX.pam4_DFE(dfe_tap_weights)

sdp.simple_eye(RX.signal[64*300:], samples_per_symbol*3, 1000, TX.UI/TX.samples_per_symbol, f"Eye Diagram with CTLE, FFE, and DFE")

#%%
np.save("./data/rx_ffe_tap_weights.npy",ffe_tap_weights)
np.save("./data/rx_dfe_tap_weights.npy",dfe_tap_weights)
np.save("./data/tx_fir_tap_weights.npy",tx_fir_tap_weights)
