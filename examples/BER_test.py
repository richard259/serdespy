import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp

h = np.load("./data/h_thru.npy")
h_ctle = np.load("./data/h_ctle.npy")
ffe_tap_weights = np.load("./data/rx_ffe_tap_weights.npy")
dfe_tap_weights = np.load("./data/rx_dfe_tap_weights.npy")
tx_fir_tap_weights = np.load("./data/tx_fir_tap_weights.npy")

nyquist_f = 26.56e9

samples_per_symbol = 64

voltage_levels = np.array([-3,-1,1,3])

data = sdp.prqs10(1)

TX = sdp.Transmitter(data, voltage_levels, nyquist_f)

TX.FIR(tx_fir_tap_weights)

TX.oversample(samples_per_symbol)

TX.gaussian_jitter(stdev_div_UI=0.025)

TX.tx_bandwidth(freq_bw=120e9)

sdp.simple_eye(TX.signal, samples_per_symbol*3, 1000, TX.UI/TX.samples_per_symbol, "TX Bandwidth-Limited Eye Diagram (-3dB frequency at 100GHz)")
#%%
signal_out = sp.signal.fftconvolve(TX.signal, h, mode = "same")

#xtalk = np.load("./data/xt_response.npy")[:384000]

#signal_out_ctle = sp.signal.fftconvolve(signal+xtalk, h_ctle, mode = "same")
signal_out_ctle = sp.signal.fftconvolve(signal_out, h_ctle, mode = "same")
#%%
sdp.simple_eye(signal_out_ctle, samples_per_symbol*3, 1000, TX.UI/TX.samples_per_symbol, "Eye Diagram with CTLE")

RX = sdp.Receiver(signal_out_ctle, samples_per_symbol, nyquist_f, voltage_levels, shift = True, main_cursor = 0.5006377086200736)

RX.slice_signal()
#%%
RX.signal_BR =  RX.signal_BR + np.random.normal(scale=0.1, size = RX.signal_BR.size)

sdp.simple_eye(RX.signal_BR, 3, 1000, TX.UI, "Eye Diagram with CTLE and noise BR")
#%%
RX.FFE_BR(ffe_tap_weights, 3)

sdp.simple_eye(RX.signal_BR, 3, 800, TX.UI/TX.samples_per_symbol, "Eye Diagram after FFE")
#%%
RX.pam4_DFE_BR(dfe_tap_weights)

sdp.simple_eye(RX.signal_BR, 3, 800, TX.UI/TX.samples_per_symbol, "Eye Diagram after DFE")

#%%
err = sdp.prqs_checker(10, data, RX.symbols_out[10:-10])

print("Bits Transmitted =", RX.symbols_out[10:-10].size*2, 'Bit Errors =', err[0])

print("Bit Error Ratio = ", err[0]/(RX.symbols_out[10:-10].size*2))