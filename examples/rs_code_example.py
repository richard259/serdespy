import serdespy as sdp
import numpy as np
import time
import skrf as rf
import scipy as sp
import matplotlib.pyplot as plt

#create data input
data_in = sdp.prbs20(1);

#encode data with RS-KP4
data_in_int = sdp.bin_seq2int_seq(data_in)

kp4 = sdp.RS_KP4()

data_in_enc_int = np.array(kp4.encode(data_in_int))

data_in_enc = sdp.int_seq2bin_seq(data_in_enc_int)

#%%
#data_out_enc = data_in_enc#
#data_out_enc_int = sdp.bin_seq2int_seq(data_out_enc)
#data_out_int = np.array(kp4.decode(data_out_enc_int)[0])
#data_out = sdp.int_seq2bin_seq(data_out_int)

#%%send through channel

#define network
network = rf.Network('./touchstone/DPO_4in_Meg7_THRU.s4p')

#set up port definition of network
port_def = np.array([[0, 1],[2, 3]])

#get TF of differential network
H,f,h,t = sdp.four_port_to_diff(network,port_def)

#Nyquist frequency
nyquist_f = 26.56e9*1.219

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
#%%

#create BR sampled TX waveform
signal_in = np.copy(sdp.nrz_input(1,data_in_enc,np.array([-0.5,0.5])))

#convolve with channel coefficients
signal_out = sp.signal.fftconvolve(signal_in, channel_coefficients)

#Set up Receiver class with BR sampled RX waveform
RX = sdp.Receiver(np.copy(signal_out),1,nyquist_T, np.array([-0.5,0.5]),shift=False)

#slice output waveform to recover data
data_out_enc = sdp.nrz_a2d(RX.signal,1,0)[n_taps_pre:-n_taps_post]

err = 0
for i in range (data_in_enc.size):
    if data_in_enc[i] != data_out_enc[i]:
        err = err + 1
        
print ("pre-fec err:", err)

#%%
data_out_enc_int = sdp.bin_seq2int_seq(data_out_enc)

data_out_int = np.array(kp4.decode(data_out_enc_int)[0])

data_out = sdp.int_seq2bin_seq(data_out_int)

#%% calculate BER post-FEC BER

print("Post-FEC: ", sdp.prbs_checker(20,data_in,data_out[10:-10])[0], "Errors in", data_out.size, "Bits Simulated")

print("BER = ", sdp.prbs_checker(20,data_in,data_out[10:-10])[0]/data_out[10:-10].size)


