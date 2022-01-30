from .reedsolo import RSCodec, ReedSolomonError
import numpy as np


#KP4 = RS (544, 514, 15)
def RS_KP4():
    return RSCodec(nsym = 30, nsize = 544, c_exp = 10 )

def RS_KR4():
    #KR4 = RS(528,514)
    return RSCodec(nsym = 14, nsize = 528, c_exp = 10 )



def bin_seq2int_seq(bin_seq):
    
    n_words = int(np.ceil(bin_seq.size/10))

    d_len = n_words * 10

    words = np.hstack(( bin_seq, np.zeros(d_len - bin_seq.size,dtype = np.uint8)))

    words = words.reshape((n_words,10))
    
    #return words

    int_seq = np.zeros(n_words).astype(int)
    
    for i in range(n_words):
        int_seq[i] = bin2int(words[i])
    
    return int_seq

def int_seq2bin_seq(int_seq):
    
    n_words = int_seq.size
    
    words_out = np.zeros((n_words,10),dtype = np.uint8)
    
    for i in range(n_words):
        words_out[i] = int2bin(int_seq[i])
    
    bin_seq = np.ndarray.flatten(words_out)
    
    return bin_seq

def bin2int(x):
    y = 0
    y += x[0]*512
    y += x[1]*256
    y += x[2]*128
    y += x[3]*64
    y += x[4]*32
    y += x[5]*16
    y += x[6]*8
    y += x[7]*4
    y += x[8]*2
    y += x[9]*1
    return y

def int2bin(x):
    y = np.zeros(10,dtype = np.uint8)
    y[0] = x // 512
    r = x % 512
    y[1] = r // 256
    r = r % 256
    y[2] = r // 128
    r = r % 128
    y[3] = r // 64
    r = r % 64
    y[4] = r // 32
    r = r % 32
    y[5] = r // 16
    r = r % 16
    y[6] = r // 8
    r = r % 8
    y[7] = r // 4
    r = r % 4
    y[8] = r // 2
    r = r % 2
    y[9] = r // 1
    return y
