
import serdespy as sdp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

data = sdp.prbs24(1)

#%% KR4 FEC
KR4_encoder = sdp.RS_KR4()

N = 528
K = 514
T = 7

n_bits_FEC_symbol = 10

n_bits_payload = K*n_bits_FEC_symbol

prob_err = 0.0005

i = 0

total_bit_error_pre_FEC= 0
total_bit_pre_FEC = 0

total_bit_error_post_FEC= 0
total_bit_post_FEC = 0

total_frame_error= 0
total_frame = 0

while i<10000:
    
    payload = data[(i%1000)*n_bits_payload:((i%1000)+1)*n_bits_payload]
    codeword = sdp.rs_encode(payload, KR4_encoder,pam4=False)
    codeword_rx = np.copy(codeword)
    
    #add errors
    for bit in range(codeword_rx.size):
        if np.random.rand() < prob_err:
            len_burst = int(np.ceil(np.random.rand()*20))
            
            for err in range(len_burst):
                if bit+err<codeword_rx.size:
                    if codeword_rx[bit+err] ==1:
                        codeword_rx[bit+err] = 0
                    elif codeword_rx[bit+err] == 0:
                        codeword_rx[bit+err] = 1
            
    payload_dec = sdp.rs_decode(codeword_rx, KR4_encoder,pam4=False)

    n_bit_errors_pre_FEC = sum(codeword!=codeword_rx)
    n_bit_errors_post_FEC = sum(payload_dec!=payload)
    
    total_bit_error_pre_FEC = total_bit_error_pre_FEC + n_bit_errors_pre_FEC
    total_bit_pre_FEC = total_bit_pre_FEC + codeword.size
    
    total_bit_error_post_FEC = total_bit_error_post_FEC + n_bit_errors_post_FEC
    total_bit_post_FEC = total_bit_post_FEC + payload.size
    
    if np.array_equal(payload_dec,payload) == False:
        total_frame_error = total_frame_error + 1
        
    total_frame = total_frame+1
    
    i = i + 1

BER_pre_FEC = total_bit_error_pre_FEC/total_bit_pre_FEC

BER_post_FEC = total_bit_error_post_FEC/total_bit_post_FEC

FER = total_frame_error/total_frame

print('KR4 - pre-FEC BER: %2f, post-FEC BER: %2f FER: %2f' % (BER_pre_FEC,BER_post_FEC,FER))

#%% KR4 FEC
KP4_encoder = sdp.RS_KP4()

N = 544
K = 514
T = 15

n_bits_FEC_symbol = 10

n_bits_payload = K*n_bits_FEC_symbol

i = 0

total_bit_error_pre_FEC= 0
total_bit_pre_FEC = 0

total_bit_error_post_FEC= 0
total_bit_post_FEC = 0

total_frame_error= 0
total_frame = 0

while i<10000:
    
    payload = data[(i%1000)*n_bits_payload:((i%1000)+1)*n_bits_payload]
    codeword = sdp.rs_encode(payload, KP4_encoder,pam4=False)
    codeword_rx = np.copy(codeword)
    
    #add errors
    for bit in range(codeword_rx.size):
        if np.random.rand() < prob_err:
            len_burst = int(np.ceil(np.random.rand()*20))
            #print(len_burst)
            
            for err in range(len_burst):
                if bit+err<codeword_rx.size:
                    if codeword_rx[bit+err] ==1:
                        codeword_rx[bit+err] = 0
                    elif codeword_rx[bit+err] == 0:
                        codeword_rx[bit+err] = 1
            
    payload_dec = sdp.rs_decode(codeword_rx, KP4_encoder,pam4=False)

    n_bit_errors_pre_FEC = sum(codeword!=codeword_rx)
    n_bit_errors_post_FEC = sum(payload_dec!=payload)
    
    total_bit_error_pre_FEC = total_bit_error_pre_FEC + n_bit_errors_pre_FEC
    total_bit_pre_FEC = total_bit_pre_FEC + codeword.size
    
    total_bit_error_post_FEC = total_bit_error_post_FEC + n_bit_errors_post_FEC
    total_bit_post_FEC = total_bit_post_FEC + payload.size
    
    if np.array_equal(payload_dec,payload) == False:
        total_frame_error = total_frame_error + 1
        
    total_frame = total_frame+1
    
    i = i + 1

BER_pre_FEC = total_bit_error_pre_FEC/total_bit_pre_FEC

BER_post_FEC = total_bit_error_post_FEC/total_bit_post_FEC

FER = total_frame_error/total_frame

print('KP4 - pre-FEC BER: %2f, post-FEC BER: %2f FER: %2f' % (BER_pre_FEC,BER_post_FEC,FER))
