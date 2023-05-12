'''
This file contains an example of encoding 1M bits with with the RS KP4 or RS KR4 error correction code
'''

import serdespy as sdp
import time

data = sdp.prbs20(1)[:1000000]

#encoder = sdp.RS_KR4()
encoder = sdp.RS_KP4()

t1 = time.time()
data_encoded = sdp.rs_encode(data, encoder, pam4=False)
t2 = time.time()

print(f'time to encode: {t2-t1}')


t1 = time.time()
data_decoded = sdp.rs_decode(data_encoded, encoder, pam4=False)
t2 = time.time()

print(f'time to decode: {t2-t1}')