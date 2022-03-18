import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt

signal = np.load("../examples/pam4_eye_ctle.npy")
                 
sorted_traces = sdp.pam4_stat_eye_new(signal, 32 , 2000, 2.941453313253012e-13, "Statistical Eye Diagram", probs = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1], vrange = [-5,5], n_symbols=3)


#%%

