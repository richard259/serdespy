import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#%%
n_pre = 0

n_post = 1

channel_coefficients = np.array([1,0])

data_in = np.load("prqs12.npy")

signal_in = np.load("prqs12_signal.npy")

voltage_levels = np.array([-3,-1,1,3])

signal_out = sp.signal.fftconvolve(signal_in,channel_coefficients)[n_pre:-n_post]

RX = sdp.Receiver(np.copy(signal_out),1,1, voltage_levels, shift=False)

noise_variance = 0.0621

RX.noise(np.sqrt(noise_variance))

data_out = sdp.pam4_a2d(RX.signal,1,voltage_levels)

err = sdp.prqs_checker(12,data_in,data_out)

print("Bits Transmitted =", data_out.size*2, 'Bit Errors =', err[0])

print("Bit Error Ratio = ", err[0]/(data_out.size*2))


#%%
n = data_out.size*2
k = err[0]

def p(x):
    return (n+1) * (np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(n-k))) *np.power(x,k)* np.power(1-x,n-k)

def stdev_integrand(x):
    return p(x)*(x-(k/n))**2

ber = np.linspace (0,1e-4,1000)

plt.plot(ber,p(ber))

#stdev = np.sqrt(sp.integrate.quad(stdev_integrand, 0, 1))

#%%
n_pre = 0

n_post = 2

channel_coefficients = np.array([1,0.55,0.3])

data_in = np.load("prqs12.npy")

data_in = np.hstack((data_in,data_in,data_in,data_in,data_in))

signal_in = np.load("prqs12_signal.npy")

signal_in = np.hstack((signal_in,signal_in,signal_in,signal_in,signal_in))

#%%
voltage_levels = np.array([-3,-1,1,3])

signal_out = sp.signal.fftconvolve(signal_in,channel_coefficients)[n_pre:-n_post]

#%%
RX = sdp.Receiver(np.copy(signal_out),1,1, voltage_levels, shift=False)

noise_variance = 0.0621

RX.noise(np.sqrt(noise_variance))

RX.pam4_DFE_BR(np.array([0.55,0.3]))

data_out = sdp.pam4_a2d(RX.signal,1,voltage_levels)

err = sdp.prqs_checker(12,data_in,data_out)

print("Bits Transmitted =", data_out.size*2, 'Bit Errors =', err[0])

print("Bit Error Ratio = ", err[0]/(data_out.size*2))