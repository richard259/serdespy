# Lossy transmission line modeling
import serdespy as sdp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#% electrical loss parameters

#All AC parameters are specified at this frequency
w_0 = 2*np.pi*10e9

#loss tangent = theta_0/(2*pi)
#theta_0 = 0.022
theta_0 = 0.01

#skin-effect scaling factor [ohms/m @ w_0]
k_r = 87

#dc resistance [ohm/m]
RDC = 0.0001

# Biuld frequency vector

#fmax=1e12 -> time step = 500fs
fmax=1e12

#frequency vector (Hz)
k = 14
f = np.linspace(0,fmax,2**k+1)

#frequency vector (rad/s)
w = f*2*np.pi

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

#Inductance [H/m]
L0 = Z0/v0

#Capacitance [F/m]
C0 = 1/(Z0*v0)

#Conductance [S/m]
G0 = 1e-12

#Resistance
RAC = (k_r*(1+1j)*np.sqrt(w/w_0))

#%% Generate frequency-dependent RLGC for the lossy transmission line
R=np.sqrt(RDC**2 + RAC**2)
L=L0*np.ones(np.size(f))
G=G0*np.ones(np.size(f))
C= C0 * (1j*w/w_0)**(-2*theta_0/np.pi)

if (f[0]==0):
   C[0] = C[1]
   
# transmission line length [m]
d = 0.1

#create transmission ABCD paramaters
tline = sdp.rlgc(R,L,G,C,d,f);

#%% source impedance
r_source = 50
source = sdp.impedance(r_source*np.ones(np.size(f)))

#termination admittance
r_term = 50
termination = sdp.admittance(np.ones(np.size(f))/r_term)

#channel is the series connection of the source, transmission line, and termination
channel = sdp.series(np.array([source,tline,termination]))

#%%
# frequency domain response
Hchannel = 1/channel[:,0,0]

#Hchannel = Hchannel/abs(Hchannel[0])

np.save("./data/Hchannel",Hchannel)

# impulse response
h,t = sdp.freq2impulse(Hchannel,f);
np.save("./data/h.npy",h)
np.save("./data/t.npy",t)

#step response
hstep = sp.signal.convolve(h,np.ones(np.shape(h)))[:np.size(h)]

#100Gbps
data_rate = 100e9

#Pam-4 Signalling
t_symbol = 2/data_rate

#time step between samples of impulse response
t_sample = 1/(2*fmax)

#number of time samples in one PAM-4 symbol
samples_per_symbol = int(t_symbol/t_sample)

#response of transmission line to one UI pulse
hpulse = sp.signal.convolve(h,np.ones(np.array([samples_per_symbol,])))[:np.size(h)]
np.save("./data/hpulse.npy",hpulse)

#%% Plots

plt.figure(dpi=600)
plt.title('Transmission Line Frequency Response')
plt.semilogx(1e-9*f,20*np.log10(np.abs(Hchannel)))
plt.xlim([0.1, 100])
plt.ylim([-40, 2])
plt.xlabel('Frequency [GHz]')
plt.ylabel('Mag Response [dB]')
plt.grid()
#nyquist f for 100G 4-PAM is 25G
plt.axvline(x=25,color = 'grey', label = "Nyquist Frequency")
plt.show()

plt.figure(dpi=600)
plt.plot((t*1e9),h)
plt.title('Transmission Line Impulse Response')
plt.ylabel('Impulse Response')
plt.xlabel('Time (ns)')
plt.xlim([0, 5])
#plt.ylim([-0.01, 0.08])
plt.show()

plt.figure(dpi=600)
plt.plot((t*1e9),hpulse)
plt.title('Transmission Line Pulse Response')
plt.ylabel('Pulse Response')
plt.xlabel('Time (ns)')
plt.xlim([0, 5])
#plt.ylim([-0.01, 0.08])
plt.show()

plt.figure(dpi=600)
plt.plot(t*1e9,hstep)
plt.title('Transmission Line Step Response')
plt.ylabel('Step Response [V]')
plt.xlabel('Time (ns)')
plt.xlim([0, 5])
plt.show()


#%% Eye Diagram

#generate binary data
data = sdp.prqs10(1)[:10000]

data_rate = 100e9

#generate Baud-Rate sampled signal from data
signal_BR = sdp.pam4_input_BR(data)

#oversampled signal
signal_ideal = np.repeat(signal_BR, samples_per_symbol)

#eye diagram of ideal signal

signal_out = sp.signal.convolve(h,signal_ideal)
                                
sdp.simple_eye(signal_out[100*samples_per_symbol:], samples_per_symbol*3, 500, t_sample, "{}Gbps 4-PAM Signal".format(data_rate/1e9))

#%% Add parallel shunt capacitance to source and termination

#200 fF capacitance
C = 200*1e-15

#ABCD paramaters of shunt cap network
cap_network = sdp.shunt_cap(C,w*1j)

#channel is the series connection of the networks
channel = sdp.series(np.array([source,cap_network,tline,cap_network,termination]))

#%%
# frequency domain response
Hchannel = 1/channel[:,0,0]

#Hchannel = Hchannel/abs(Hchannel[0])

np.save("./data/Hchannel_cap",Hchannel)

# impulse response
h,t = sdp.freq2impulse(Hchannel,f);

#step response
hstep = sp.signal.convolve(h,np.ones(np.shape(h)))[:np.size(h)]

#100Gbps
data_rate = 100e9

#Pam-4 Signalling
t_symbol = 2/data_rate

#time step between samples of impulse response
t_sample = 1/(2*fmax)

#number of time samples in one PAM-4 symbol
samples_per_symbol = int(t_symbol/t_sample)

#response of transmission line to one UI pulse
hpulse = sp.signal.convolve(h,np.ones(np.array([samples_per_symbol,])))[:np.size(h)]
np.save("./data/hpulse_cap.npy",hpulse)

#%% Plots

plt.figure(dpi=600)
plt.title('Transmission Line (with cap) Frequency Response Magnitude')
plt.semilogx(1e-9*f,20*np.log10(np.abs(Hchannel)))
plt.xlim([0.1, 100])
plt.ylim([-40, 2])
plt.xlabel('Frequency [GHz]')
plt.ylabel('Mag Response [dB]')
plt.grid()
#nyquist f for 100G 4-PAM is 25G
plt.axvline(x=25,color = 'grey', label = "Nyquist Frequency")
plt.show()


plt.figure(dpi=600)
plt.plot((t*1e9),h)
plt.title('Transmission Line (with cap) Impulse Response')
plt.ylabel('Impulse Response')
plt.xlabel('Time (ns)')
plt.xlim([0, 5])
#plt.ylim([-0.01, 0.08])
plt.show()

plt.figure(dpi=600)
plt.plot((t*1e9),hpulse)
plt.title('Transmission Line (with cap) Pulse Response')
plt.ylabel('Pulse Response')
plt.xlabel('Time (ns)')
plt.xlim([0, 5])
#plt.ylim([-0.01, 0.08])
plt.show()

plt.figure(dpi=600)
plt.plot(t*1e9,hstep)
plt.title('Transmission Line (with cap) Step Response')
plt.ylabel('Step Response [V]')
plt.xlabel('Time (ns)')
plt.xlim([0, 5])
plt.show()


#%% Eye Diagram
#eye diagram of ideal signal

signal_out_cap = sp.signal.convolve(h,signal_ideal)
                                
sdp.simple_eye(signal_out_cap[100*samples_per_symbol:], samples_per_symbol*3, 500, t_sample, "{}Gbps 4-PAM Signal".format(data_rate/1e9))

#%% save data for next homework assignment
np.save("./data/signal.npy",signal_out)
np.save("./data/signal_cap.npy",signal_out_cap)
np.save("./data/f.npy",f)
np.save("./data/w.npy",w)
