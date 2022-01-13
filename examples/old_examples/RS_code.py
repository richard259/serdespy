# Initialization
from reedsolo import RSCodec, ReedSolomonError
import serdespy as sdp
import numpy as np


#KP4 = RS (544, 514, 15)
rsc = RSCodec(nsym = 30, nsize = 544, c_exp = 10 )

# KR4 = RS(528,514)
#rsc = RSCodec(nsym = 14, nsize = 528, c_exp = 10 )

data_in = sdp.prbs13(1);

n_words = int(np.ceil(data_in.size/10))

d_len = n_words * 10

words_in = np.hstack(( data_in, np.zeros(d_len - data_in.size,dtype = np.uint8)))

words_in = words_in.reshape((n_words,10))

def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

words_in_decimal = np.zeros(n_words)

for i in range(n_words):
    words_in_decimal[i] = bool2int(words_in[i])

n_symbols = int(np.ceil(n_words/514))

symbols_len = n_symbols * 514

symbols_in = np.hstack((words_in_decimal, np.zeros(symbols_len - words_in_decimal.size)))

symbols_in = symbols_in.reshape((n_symbols,514))

encoded = np.zeros((n_symbols, 544))


#%%
for i in range(n_symbols):
    encoded_bytes = rsc.encode(str(np.ndarray.tobytes(symbols_in[i])))
    encoded_ints = np.frombuffer(encoded_bytes, count = 10 dtype=int)
    