# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 18:20:45 2021

@author: Richard Barrie
"""

from functions import *
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

def fourPortToDiffXtalk(THRU, FEXT, NEXT, port_def):
    
    #number of input signals to the network
    n_inputs = 1 + len(NEXT) + len(FEXT)
    
    #construct A matrix
    A = np.array([[1,-1]])
    
    #construct B matrix
   
    #thru networks
    ch1_thru = rf.subnetwork(THRU, [port_def[0][0], port_def[0][1]])
    ch1_thru_abcd = rf.s2a(ch1_thru.s)

    ch2_thru = rf.subnetwork(THRU, [port_def[1][0], port_def[1][1]])
    ch2_thru_abcd = rf.s2a(ch2_thru.s)
    
    
    #xtalk networks
    tx1_rx2 = rf.subnetwork(THRU, [port_def[0][0], port_def[1][1]])
    tx1_rx2_abcd = rf.s2a(tx1_rx2.s)
    
    tx2_rx1 = rf.subnetwork(THRU, [port_def[0][1], port_def[1][0]])
    tx2_rx1_abcd = rf.s2a(tx2_rx1.s)

    #add termination to match charictaristic impedance of networks 12 and 34
    term1 = admittance(1/ch1_thru.z0[:,0])
    ch1_thru_abcd = series(ch1_thru_abcd,term1)

    term2 = admittance(1/ch2_thru.z0[:,0])
    ch2_thru_abcd = series(ch2_thru_abcd,term2)
    
    #get discrete transfer function for subnetworks, assuming 0 source impedance
    H1_thru = 1/ch1_thru_abcd[:,0,0]
    H2_thru = 1/ch2_thru_abcd[:,0,0]
    H_tx1_rx2 = 1/tx1_rx2_abcd[:,0,0]
    H_tx2_rx1 = 1/tx2_rx1_abcd[:,0,0]
    
    
    B  = np.zeros((ch1_thru_abcd.shape[0],2,2),dtype=complex)
    
    B[:,0,0] = H1_thru
    B[:,0,1] = H_tx1_rx2
    B[:,1,0] = H_tx2_rx1
    B[:,1,1] = H2_thru
    
    #B = np.array([[H1_thru, H_tx2_rx1],[H_tx2_rx1, H2_thru]])
    thru_TF = B
    
   # print('here',B.shape, fourPortSubnetworks(FEXT[0],port_def).shape)
    
    for i in range(len(FEXT)):
        B = np.concatenate((B,fourPortSubnetworks(FEXT[i],port_def)),axis=2)
        
    for i in range(len(NEXT)):
        B = np.concatenate((B,fourPortSubnetworks(NEXT[i],port_def)),axis=2)
        
    #construct C matrix
    
    for i in range(len(NEXT)):
        if (i==0):
            C = thru_TF
        else:
            C = block_diag_3d(C, thru_TF)
    
    
    Id = Id3(ch1_thru_abcd.shape[0],len(FEXT)*2+2)
    C = block_diag_3d(Id,C)
    
    #construct D matrix
    
    for i in range(n_inputs):
        if (i==0):
            D = np.array([[0.5],[-0.5]])
        else:
            D = sp.linalg.block_diag(D, np.array([[0.5],[-0.5]]))
        
        
   # print ('A =',A.shape)
   # print ('B =',B.shape)
   # print ('C =',C.shape)
   # print ('D =',D.shape)
    
    
    return A@B@C@D, THRU.f

def block_diag_3d(A,B):
    if A.shape[0] != B.shape[0]:
        print('blockdiag error')
        return False
    
    C = np.zeros((A.shape[0],A.shape[1]+B.shape[1],A.shape[2]+B.shape[2]),dtype=complex)
    
    C[:, 0:A.shape[1] , 0:A.shape[2] ] = A
    
    C[:,A.shape[1]:A.shape[1]+B.shape[1], A.shape[2]:A.shape[2]+B.shape[2]] = B
    
    return C
    
def Id3(n,d):
    out = np.zeros((n,d,d))
    for i in range(d):
        out[:,i,i] = np.ones((n))
    return out

def fourPortSubnetworks (network, port_def):
    
    #thru networks
    net01 = rf.subnetwork(network, [port_def[0][0], port_def[0][1]])
    net01_abcd = rf.s2a(net01.s)

    net23 = rf.subnetwork(network, [port_def[1][0], port_def[1][1]])
    net23_abcd = rf.s2a(net23.s)
    
    
    #xtalk networks
    net03 = rf.subnetwork(network, [port_def[0][0], port_def[1][1]])
    net03_abcd = rf.s2a(net03.s)
    
    net21 = rf.subnetwork(network, [port_def[0][1], port_def[1][0]])
    net21_abcd = rf.s2a(net21.s)
    
    out  = np.zeros((net01_abcd.shape[0],2,2),dtype=complex)
    
    out[:,0,0] = 1/net01_abcd[:,0,0]
    out[:,0,1] = 1/net21_abcd[:,0,0]
    out[:,1,0] = 1/net03_abcd[:,0,0]
    out[:,1,1] = 1/net23_abcd[:,0,0]
    
    return out

FE =  rf.Network('./DPO_4in_Meg7_FENH89.s4p')
NE =  rf.Network('./DPO_4in_Meg7_NENF89.s4p')
THRU =  rf.Network('./DPO_4in_Meg7_THRU.s4p')

port_def = np.array([[0, 1],[2, 3]])

Xtalk_network,f = fourPortToDiffXtalk(THRU, [FE], [NE], port_def)

H_THRU = Xtalk_network[:,0,0]
H_FE = Xtalk_network[:,0,1]
H_NE = Xtalk_network[:,0,2]



h_THRU,t = freq2impulse(H_THRU,f)
h_FE,t = freq2impulse(H_FE,f)
h_NE,t = freq2impulse(H_NE,f)
#%%
plt.figure(1)
plt.title('Impulse Response THRU')
plt.plot(t*1e9,h_THRU)
plt.ylabel('Impulse Response')
plt.xlabel('Time (ns)')

plt.figure(2)
plt.title('Impulse Response FEXT-victim')
plt.plot(t*1e9,h_FE)
plt.ylabel('Impulse Response')
plt.xlabel('Time (ns)')

plt.figure(3)
plt.title('Impulse Response NEXT-victim')
plt.plot(t*1e9,h_NE)
plt.ylabel('Impulse Response')
plt.xlabel('Time (ns)')

#%%
#hTHRUstep = sp.signal.fftconvolve(h_THRU,np.ones(h_THRU.size))
#hTHRUstep = hTHRUstep[0:h.size]

#plt.figure(4)
#plt.title('Step Response ')
#plt.plot(t*1e9,hstep)
#plt.ylabel('[V]')
#plt.xlabel('Time (ns)')


nyquist_f = 26.56e9

n = 16

f_max_d = 2*n*nyquist_f

t_d = 1/f_max_d

HTHRU_0, fTHRU_0, hTHRU_0, tTHRU_0 = zeroPad(H_THRU,f,t_d)
HNEXT_0, fNEXT_0, hNEXT_0, tNEXT_0 = zeroPad(H_NE,f,t_d)
HFEXT_0, fFEXT_0, hFEXT_0, tFEXT_0 = zeroPad(H_FE,f,t_d)

sim_freq = nyquist_f/4

V1_in = pam4Input(tTHRU_0,sim_freq,np.array([0,1,2,3]))
V2_in = pam4Input(tTHRU_0,sim_freq,np.array([0,1,2,3]))
V3_in = pam4Input(tTHRU_0,sim_freq,np.array([0,1,2,3]))

V1out = sp.signal.fftconvolve(hTHRU_0,V1_in)
V2out = sp.signal.fftconvolve(hFEXT_0,V2_in)
V3out = sp.signal.fftconvolve(hNEXT_0,V3_in)

signal_out = V1out+V2out+V3out
signal_out = V2out

signal_out = signal_out[0:hTHRU_0.size]

data_out = signal_out[64000:]

plt.figure(4)
plt.title('Signal out')
plt.plot(tTHRU_0*1e9,signal_out)
plt.ylabel('(V)')
plt.xlabel('Time (ns)')
#plt.xlim([0, 10])
#plt.ylim([-0.1, 0.5])


eyeDiagram(data_out, 64*4 ,400, tTHRU_0[1])

