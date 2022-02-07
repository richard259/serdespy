import serdespy as sdp
import skrf as rf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#%% #encode data with RS
data_in = sdp.prbs20(1)[:-1]

#format data into 10-bit words
data_in_int = sdp.bin_seq2int_seq(data_in)

#create KR4 encoder instance
#kp4 = sdp.RS_KR4()
rsc = sdp.RSCodec(nsym = 8, nsize = 544, c_exp = 10 )

#encode_data
data_in_enc_int = np.array(rsc.encode(data_in_int))

#convert back to binary sequence
data_in_enc = sdp.int_seq2bin_seq(data_in_enc_int)

#convert into pam4 symbols
len_symbols_enc = int(data_in_enc.size/2)

data_in_enc_reshape = np.reshape(data_in_enc,(2,len_symbols_enc) , order = 'F')

symbols_in_enc = np.zeros(len_symbols_enc,dtype = np.uint8)

for i in range(len_symbols_enc):
    symbols_in_enc[i] = sdp.grey_code(data_in_enc_reshape[:,i])

#%%
voltage_levels = np.array([-3,-1,1,3])


#convert pam4 symbols into Baud-rate sampled ideal waveform
signal_in_enc = sdp.pam4_input(1, symbols_in_enc, voltage_levels)


#convolve with channel coef
#channel_coefficients = np.array([1,0.55,0.3])
channel_coefficients = np.array([0.6,0.2,-0.2])

signal_out_enc = sp.signal.fftconvolve(signal_in_enc,channel_coefficients)

RX = sdp.Receiver(signal_out_enc[:-2],1,1, voltage_levels, shift=False, main_cursor=0.6)

noise_variance = 0.03


#%%

#noise_variances = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

noise_variances = [0.024, 0.023, 0.022]

#noise_variances = [0.038, 0.035, 0.032]

for noise_variance in noise_variances:
    
    n_bits = 0
    n_errors = 0
    
    while True:
        RX.reset()
        RX.noise(np.sqrt(noise_variance))
        #RX.pam4_DFE_BR(np.array([0.55,0.3]))
        
        RX.pam4_DFE_BR(np.array([0.2,-0.2]))
        
        symbols_out_enc = RX.symbols_out
        
        #sdp.pam4_a2d(RX.signal,1,voltage_levels*0.6)
        
        data_out_enc = np.zeros(symbols_out_enc.size*2,dtype = np.uint8)
        
        for i in range(symbols_out_enc.size):
            #print(i,symbols_out[i],data_out[:10])
            data_out_enc[i*2:i*2+2] = sdp.grey_decode(symbols_out_enc[i])
        
        data_out_enc_int = sdp.bin_seq2int_seq(data_out_enc)
        
        data_out_int = np.array(rsc.decode(data_out_enc_int)[0])
        
        data_out = sdp.int_seq2bin_seq(data_out_int)
        
        for i in range(data_in.size):
            if data_in[i] != data_out[i]:
                n_errors = n_errors + 1
        
        n_bits = n_bits + data_out.size
        
        print("errors = ", n_errors, "n_bits = ", n_bits)
        
        if n_errors > 400:
            print('var = ', noise_variance)
            print("Bits Transmitted =", n_bits, 'Bit Errors =', n_errors )
            print("Bit Error Ratio = ", n_errors/n_bits)
            break
