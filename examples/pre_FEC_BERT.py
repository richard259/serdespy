import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

n_bits = 0
n_errors = 0
#%%
data_in = sdp.prbs20(1)[:-1]

data_in_reshape = np.reshape(data_in,(2,2**19-1),order = 'F')

symbols_in = np.zeros(2**19-1,dtype = np.uint8)

for i in range(2**19-1):
    symbols_in[i] = sdp.grey_code(data_in_reshape[:,i])
    
voltage_levels = np.array([-3,-1,1,3])

signal_in = sdp.pam4_input(1, symbols_in, voltage_levels)

n_pre = 0

n_post = 2

#channel_coefficients = np.array([1,0.55,0.3])
channel_coefficients = np.array([0.6,0.2,-0.2])


#%%
signal_out = sp.signal.fftconvolve(signal_in,channel_coefficients)

RX = sdp.Receiver(signal_out[:-2],1,1, voltage_levels, shift=False, main_cursor=0.6)

#noise_variance = 0.03


#%%


#noise_variances = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

#noise_variances = [0.029, 0.028, 0.027, 0.026, 0.025, 0.024, 0.023, 0.022]

noise_variances = [0.022]

for noise_variance in noise_variances:
    
    n_bits = 0
    n_errors = 0
    
    while True:
        RX.reset()
        RX.noise(np.sqrt(noise_variance))
        RX.pam4_DFE_BR(np.array([0.2,-0.2]))
        
        symbols_out = sdp.pam4_a2d(RX.signal,1,voltage_levels*0.6)
        
        data_out = np.zeros(data_in.size,dtype = np.uint8)
        
        for i in range(symbols_out.size):
            #print(i,symbols_out[i],data_out[:10])
            data_out[i*2:i*2+2] = sdp.grey_decode(symbols_out[i])
        
        for i in range(data_out.size):
            if data_in[i] != data_out[i]:
                n_errors = n_errors + 1
        
        n_bits = n_bits + data_out.size
        
        print("errors = ", n_errors, "n_bits = ", n_bits)
        
        if n_errors > 500:
            print('var = ', noise_variance)
            print("Bits Transmitted =", n_bits, 'Bit Errors =', n_errors )
            print("Bit Error Ratio = ", n_errors/n_bits)
            break
