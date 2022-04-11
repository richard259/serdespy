import serdespy as sdp
import time

data = sdp.prbs20(1)[:1000000]

encoder = sdp.RSCodec(nsym = 8, nsize = 544, c_exp = 10)
#encoder = sdp.RS_KR4()

t1 = time.time()
data_encoded = sdp.rs_encode(data, encoder, pam4=False)
t2 = time.time()

print(f'time to encode: {t2-t1}')


t1 = time.time()
data_decoded = sdp.rs_decode(data_encoded, encoder, pam4=False)
t2 = time.time()

print(f'time to decode: {t2-t1}')