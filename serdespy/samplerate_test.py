"""
Example of creating TX waveform, and creating transmission line
"""

import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

#%% set up some parameters for the transmitter waveform

voltage_levels = np.array([-3,-1,1,3])
samples_per_symbol=128
#nyquist_f = 26.56e9

#frequency is 5G 
nyquist_f = 5e9

#%% Create data from PRQS
data = sdp.prqs10(1)[:10000]

#%% Create transmitter class
time1=time.time()
TX = sdp.Transmitter(data, voltage_levels, nyquist_f)
time2=time.time()
print('time to make input for 2M bit signal: ', time2-time1)

#%%Oversample the TX waveform
time1=time.time()
TX.new_oversample(samples_per_symbol)
time2=time.time()
print('time to oversample: ', time2-time1)

#%%Add Jitter to TX waveform
time1=time.time()
TX.gaussian_jitter()
time2=time.time()
print('time to add jitter: ', time2-time1)

#%%Apply Bandwith Limitation
TX.tx_bandwidth(freq_bw=50e9)

#%% Eye diagram of over-sampled TX wf
sdp.simple_eye(TX.signal, samples_per_symbol*3, 500, TX.UI/TX.samples_per_symbol, "TX Eye Diagram")

#%%Downsample by factor of 16
TX.new_downsample(16)

#%%
sdp.simple_eye(TX.signal_downsampled, TX.samples_per_symbol*3, 500, TX.UI/TX.samples_per_symbol, "TX Eye Diagram Downsampled to 8 samples per UI")

#%% Create a transmission line from s-params

#% electrical loss parameters

#All AC parameters are specified at this frequency
w_0 = 2*np.pi*10e9

#loss tangent = theta_0/(2*pi)
theta_0 = 0.022

#skin-effect scaling factor [ohms/m @ w_0]
k_r = 87

#dc resistance [ohm/m]
RDC = 0.0001

# Biuld frequency vector

#length of frequency vector
k = 14

#fmax=100e9 -> Ts=5ps

fmax = 100e9

#frequency vector (Hz)
f = np.linspace(0,fmax,2**k+1)

#frequency vector (rad/s)
w = 2*np.pi*f

# Constants

#speed of light [m/s]
c = 2.998e8

#Vacuum permittivity [F/m]
eps_0 = 8.85*1e-12

# Transmission line parameters

#Effective relative dielectric constant
eps_r = 4.9

#Propagation velocity of the transmission line [m/s]
v0 = np.sqrt(1/eps_r)*c

#Characteristic impedance [ohm]
Z0 = 50

# Transmission line parameters

#Inductance [H/m]
L0 = Z0/v0

#Capacitance [F/m]
C0 = 1/(Z0*v0)

#Conductance [S/m]
G0 = 1e-12

#Resistance
RAC = (k_r*(1+1j)*np.sqrt(w/w_0))

# Generate frequency-dependent RLGC for the lossy transmission line
R=np.sqrt(RDC**2 + RAC**2)
L=L0*np.ones(np.size(f))
G=G0*np.ones(np.size(f))
C= C0 * (1j*w/w_0)**(-2*theta_0/np.pi)

if (f[0]==0):
   C[0] = C[1]
   
# transmission line length [m]
d = 0.1

#create transmission line model
tline = sdp.rlgc(R,L,G,C,d,f);

#termination does not perfectly match Z0

termination = sdp.admittance(np.ones(np.size(f))/60)

channel = sdp.series(tline, termination)

# frequency domain response
Hchannel = 1/channel[:,0,0]
Hchannel_dB = 20*np.log10(np.abs(Hchannel))

#find 3dB frequency
for i in range(2**k+1):
    if Hchannel_dB[i] < -3:
        f_3dB = 1e-9*f[i]
        print(f"3dB frequency of channel: {f_3dB} GHz")
        break

#time domain response
h_channel, t_channel = sdp.freq2impulse(Hchannel,f);

#bode plot of tline
plt.figure()
plt.title('Transmission Line Response')
plt.semilogx(1e-9*f,Hchannel_dB)
plt.axvline(x = f_3dB )
plt.xlabel('Frequency [GHz]')
plt.ylabel('Mag Response [dB]')

#%% resample impulse response so that timestep matches TX waveform

import samplerate

t_channel_resample = samplerate.resample(t_channel, t_channel[1]/(TX.UI/TX.samples_per_symbol), 'zero_order_hold')

h_channel_resample = samplerate.resample(h_channel, t_channel[1]/(TX.UI/TX.samples_per_symbol), 'zero_order_hold')


h = h_channel_resample[50:200]

plt.figure()
plt.title('Transmission Line Impulse Response')
plt.plot(1e9*t_channel_resample[:150],h)
plt.xlabel('Time [ns]')
plt.ylabel('Impulse Response')


# %% Convolve TX wf with channel response and plot eye diagram
rx_signal = sp.signal.fftconvolve(h, TX.signal_downsampled)

RX = sdp.Receiver(rx_signal, 8, t_channel_resample[1], voltage_levels)

sdp.simple_eye(RX.signal, RX.samples_per_symbol*3, 500, RX.t_step, "RX Eye Diagram")

