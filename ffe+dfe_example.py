"""Example of FFE operation"""

import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#define network
network = rf.Network('./DPO_4in_Meg7_THRU.s4p')

#set up port definition of network
port_def = np.array([[0, 1],[2, 3]])

#get TF of differential network
H,f,h,t = sdp.four_port_to_diff(network,port_def)

#Nyquist frequency
nyquist_f = 26.56e9

#Period of clock at nyquist frequency
nyquist_T = 1/nyquist_f

#desired number of samples per clock period
n = 64

#timesteps per bit
steps_per_symbol = int(round(n/2))

#Desired time-step
t_d = nyquist_T/n

#compute response of zero-padded TF
H, f, h, t = sdp.zero_pad(H,f,t_d)



#%% mesauure precursor and postcursor from pulse response

n_taps_post = 1
n_taps_pre = 1
n_taps = n_taps_pre+n_taps_post+1
cursor = np.zeros(n_taps)
t_vec = np.zeros(n_taps_pre+n_taps_post+1)
xcoords = []
half_symbol = int(round(n/4))

#create pulse waveform
pulse_input = np.hstack((np.ones(steps_per_symbol),np.zeros(t.size-steps_per_symbol)))

#compute pulse response
pulse_response = sp.signal.fftconvolve(h, pulse_input)
pulse_response = pulse_response[0:t.size]


#find peak of pulse response
max_idx = np.where(pulse_response == np.amax(pulse_response))[0][0]


for tap in range(-n_taps_pre,n_taps_post+1):
    #print(tap)
    #print('start',tap*steps_per_symbol-half_symbol)
    #cursor[tap+n_taps_pre] = np.average(pulse_response[max_idx+tap*steps_per_symbol-half_symbol:max_idx+(tap+1)*steps_per_symbol-half_symbol])
    cursor[tap+n_taps_pre] = pulse_response[max_idx+tap*steps_per_symbol]
    #print('tw',tap_weights[i])
    xcoords = xcoords + [1e9*t[max_idx+tap*steps_per_symbol-half_symbol]]
    t_vec[tap+n_taps_pre] = t[max_idx + tap*steps_per_symbol]

xcoords = xcoords + [1e9*t[max_idx+(n_taps_post+1)*steps_per_symbol-half_symbol]]


#plot pulse response and cursor samples
plt.figure()
plt.plot(t_vec*1e9, cursor, 'o',label = 'Cursor samples')
plt.plot(t*1e9,pulse_response, label = 'Pulse Response')
plt.xlabel("Time [ns]")
plt.ylabel("[V]")
plt.xlim([1.8,2.1])
plt.title("Tap Weight Estimation From Pulse Response")
plt.legend()

for xc in xcoords:
    plt.axvline(x=xc,color = 'grey')
#%% solve for ideal FFE tap weights
    
A = np.zeros((n_taps,n_taps))

#print(cursor)

for i in range(n_taps):
   # print (i,n_taps-abs(i-n_taps_pre),i-n_taps_pre)
    #print (np.ones(n_taps-(i-n_taps_pre))*cursor[i])
    #print(np.diag(np.ones(n_taps-abs(i-n_taps_pre))*cursor[i],(i-n_taps_pre) ))
    A += np.diag(np.ones(n_taps-abs(i-n_taps_pre))*cursor[i],(n_taps_pre-i) )

c = np.zeros((n_taps,1))
c[n_taps_pre] = 1

b = np.linalg.inv(A)@c

b = b/np.sum(abs(b))

ffe_tap_weights = b.T

#%%apply our FFE to pulse response signal

voltage_levels = np.array([-0.5, 0.5])
pulse_sig = sdp.Signal(pulse_response, steps_per_symbol, t[1], voltage_levels, shift = False)

plt.plot(t*1e9,pulse_sig.signal, label = 'Pulse Response')

pulse_sig.FFE(ffe_tap_weights,1)

#find peak of pulse response
max_idx = np.where(pulse_sig.signal == np.amax(pulse_sig.signal))[0][0]

#number of DFE taps
n_taps = 6

dfe_tap_weights = np.zeros(n_taps)
pc = np.zeros(n_taps)
xcoords = []

#Estimate tap weights based on average value of each postcursor
for i in range(n_taps):
    xcoords = xcoords + [t[max_idx-half_symbol+i*steps_per_symbol]]
    dfe_tap_weights[i] = np.average(pulse_sig.signal[max_idx+half_symbol+(i)*steps_per_symbol:max_idx+half_symbol+(i+1)*steps_per_symbol])
    pc[i] = t[max_idx +(i+1)*steps_per_symbol]
xcoords = xcoords + [t[max_idx+half_symbol+i*steps_per_symbol]]

#plot DFE tap weights
plt.plot(t*1e9,pulse_sig.signal, label = 'Pulse Response with FFE')
plt.plot(pc*1e9, dfe_tap_weights, 'o',label = 'DFE Tap Weights')
plt.legend()
plt.xlim([1.8,2.1])
for xc in xcoords:
    plt.axvline(x=xc*1e9,color = 'grey')

#%%create TX waveform

#compute input data using PRBS13
data_in = sdp.prbs13(1)

#define voltage levels for 0 and 1 bits
voltage_levels = np.array([-0.5, 0.5])

#convert data_in to time domain signal
signal_in = sdp.nrz_input(steps_per_symbol, data_in, voltage_levels)
#%%compute channel response to signal_in

#TODO: figure out convolution with different length vectors

h_zero_pad = np.hstack((h, np.zeros(signal_in.size-h.size)))

#do convolution to get differential channel response
signal_output = sp.signal.fftconvolve(h_zero_pad, signal_in)
signal_output = signal_output[0:h_zero_pad.size]

#define signal object for this signal, crop out first bit of signal which is 0 due to channel latency

sig = sdp.Signal(signal_output[5000:], steps_per_symbol, t[1], voltage_levels)
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "Eye Diagram")

#apply FFE only and show eye diagram
sig.FFE(ffe_tap_weights,1)
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "Eye Diagram with FFE")

#apply DFE and show eye diagram
sig.DFE(dfe_tap_weights,0)
sdp.simple_eye(sig.signal, sig.steps_per_symbol*2, 1000, sig.t_step, "Eye Diagram with FFE and DFE")

#%%measure BER for different levels of noise

noise_levels = np.linspace(0.03,0.05,10)
BER = np.zeros(10)

for i in range(noise_levels.size):
    sig.noise(noise_levels[i])
    sig.FFE(ffe_tap_weights,1)
    sig.DFE(dfe_tap_weights,0)
    data = sdp.nrz_a2d(sig.signal, steps_per_symbol, 0)
    errors = sdp.prbs_checker(13,data_in, data)
    BER[i] = errors[0]/data.size

#plot BER vs noise level
plt.figure()
plt.plot(noise_levels, BER, 'o')
plt.xlabel("noise stdev [V]")
plt.ylabel("BER")
plt.title("BER vs. RX noise stdev with DFE")

