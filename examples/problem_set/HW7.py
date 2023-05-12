import serdespy as sdp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


#generate binary data
data = sdp.prqs10(1)[:10000]
     
#generate Baud-Rate sampled signal from data
voltage_levels=np.array([-1,-1/3,1/3,1])
#voltage_levels=np.array([-3/2,-1/2,1/2,3/2])
#voltage_levels=np.array([-2,-2/3,2/3,2])

signal_BR = sdp.pam4_input_BR(data,voltage_levels=voltage_levels)
    
#data rate in Gbps
data_rate = 100e9

#time per 4-PAM symbol
UI = 2/data_rate

#define oversample ratio
samples_per_symbol = 64

#timestep
dt = UI/samples_per_symbol

#oversampled signal
signal_ideal = np.repeat(signal_BR, samples_per_symbol)

#eye diagram of ideal signal
sdp.simple_eye(signal_ideal, samples_per_symbol*3, 100, dt, "{}Gbps 4-PAM Signal".format(data_rate/1e9),linewidth=1.5)

#cutoff frequency
freq_bw = 50e9
       
#max frequency for constructing discrete transfer function
max_f = 1/dt

#max_f in rad/s
max_w = max_f*2*np.pi

#heuristic to get a reasonable impulse response length
ir_length = int(4/(freq_bw*dt))

#calculate discrete transfer function of low-pass filter with pole at freq_bw
w, H = sp.signal.freqs([freq_bw*(2*np.pi)], [1,freq_bw*(2*np.pi)], np.linspace(0,0.5*max_w,ir_length*4))

#frequency in hz
f = w/(2*np.pi)

#plot frequency response of TF
plt.figure(dpi=800)
plt.semilogx(1e-9*f,20*np.log10(abs(H)))
plt.ylabel('Mag. Response [dB]')
plt.xlabel('Frequency [GHz]')
plt.title("Low Pass Filter with {}GHz Cutoff Magnitude Bode Plot".format(round(freq_bw*1e-9)))
plt.grid()
plt.axvline(x=1e-9*freq_bw,color = 'grey')
plt.show()

#find impluse response of low-pass filter
h, t = sdp.freq2impulse(H,f)

#convolution of impulse response with ideal signal
signal_filtered = sp.signal.fftconvolve(signal_ideal, h[:ir_length])

#plot eye diagram of filtered signal
sdp.simple_eye(signal_filtered[samples_per_symbol*100:], samples_per_symbol*3, 100, UI/samples_per_symbol, "{}Gbps 4-PAM Signal with {}GHz Cutoff Filter".format(round(data_rate/1e9),round(freq_bw*1e-9)))

#optical modulator nonlinearity
def optical_nonlinearity(signal):
    return np.sin(np.pi*signal/5)

signal_optical = optical_nonlinearity(signal_filtered)

#eye diagram of optical signal
sdp.simple_eye(signal_optical[samples_per_symbol*100:], samples_per_symbol*3, 100, UI/samples_per_symbol, "{}Gbps Optical 4-PAM Signal".format(round(data_rate/1e9),round(freq_bw*1e-9)))

#calculate RLM
levels_optical = optical_nonlinearity(voltage_levels)

Vmin = (levels_optical[0]+levels_optical[3])/2

ES1 = (levels_optical[1]-Vmin)/(levels_optical[0]-Vmin)

ES2 = (levels_optical[2]-Vmin)/(levels_optical[3]-Vmin)

RLM = min((3*ES1),(3*ES2),(2-3*ES1),(2-3*ES2))

print("RLM = ",RLM)