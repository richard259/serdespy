# Initialization
from reedsolo import RSCodec, ReedSolomonError
import serdespy as sdp
import numpy as np


#KP4 = RS (544, 514, 15)
rsc = RSCodec(nsym = 30, nsize = 544, c_exp = 10 )
#%%
# KR4 = RS(528,514)
#rsc = RSCodec(nsym = 14, nsize = 528, c_exp = 10 )

data_in = sdp.prbs13(1);

n_words = int(np.ceil(data_in.size/10))

d_len = n_words * 10

words_in = np.hstack(( data_in, np.zeros(d_len - data_in.size,dtype = np.uint8)))

words_in = words_in.reshape((n_words,10))


#%%
#def bool2int(x):
#    y = 0
#    for i,j in enumerate(x):
#        y += j<<i
#    return y
#%%

def bool2int(x):
    y = 0
    y += x[0]*512
    print(y)
    y += x[1]*256
    print(y)
    y += x[2]*128
    print(y)
    y += x[3]*64
    print(y)
    y += x[4]*32
    print(y)
    y += x[5]*16
    print(y)
    y += x[6]*8
    print(y)
    y += x[7]*4
    print(y)
    y += x[8]*2
    print(y)
    y += x[9]*1
    print(y)
    return y

def int2bool(x):
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

#%%
bool2int(np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 1]))
#%%
data_in = sdp.prbs20(1);

n_words = int(np.ceil(data_in.size/10))

d_len = n_words * 10

words_in = np.hstack(( data_in, np.zeros(d_len - data_in.size,dtype = np.uint8)))

words_in = words_in.reshape((n_words,10))


words_in_decimal = np.zeros(n_words).astype(int)

for i in range(n_words):
    words_in_decimal[i] = bool2int(words_in[i])



#%%

t0 = time.time()
encoded = rsc.encode(words_in_decimal)
t1 = time.time()

print("time RS encode 1M bits=", t1-t0)
#%%

t0 = time.time()
decoded = np.array(rsc.decode(encoded)[0])
t1 = time.time()

print("Time to RS decode 1M bits =", t1-t0)


#%%
words_out = np.zeros(words_in.shape,dtype = np.uint8)

t0 = time.time()

for i in range(n_words):
    words_out[i] = int2bool(decoded[i])

t1 = time.time()

print("Time to put 1M bits from 10 bit ints to binary =", t1-t0)
