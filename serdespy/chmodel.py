"""Functions for Channel Modelling

Notes
-----
Based off of https://github.com/tchancarusone/Wireline-ChModel-Matlab
"""

#TODO: change comments to docstring

import numpy as np

def rlgc (r,l,g,c,d,f):
    #r, l, g, and c are transmission line paramaters:
    #r = resistance along the line per unit length [Ω/m]
    #l = inductance along the line per unit length [H/m]
    #g = conductance shunting the line per unit length [S/m]
    #c = capacitance shunting the line per unit length [F/m]
    #can be scalars or arrays if frequency dependent
    
    #l is the length of the frequency array
    #returns array s of size l*2*2 where s[n] = is the 2*2 paramater matrix
    #for the transmission line at frequency f[n]
    
    w = 2*np.pi*f
    gammad = d*np.sqrt( np.multiply( (r+1j * np.multiply(w,l)),(g+ 1j * np.multiply(w,c)) ) );
    z0 = np.sqrt( np.divide((r + 1j*np.multiply(w,l) ),(g+1j*np.multiply(w,c)))  );
    A = np.cosh(gammad)
    B = np.multiply(z0,np.sinh(gammad))
    C = np.divide(np.sinh(gammad),z0)
    D = A
    
    s = np.zeros((f.size,2,2),dtype=np.complex_)
    s[:,0,0] = A
    s[:,0,1] = B   
    s[:,1,0] = C
    s[:,1,1] = D
    
    return s

def impedance (z):
    
    #for 2-port network with impedance z (z is array if frequency dependent)
    #returns array s of size l*2*2 where s[n] = is the 2*2 paramater matrix
    #for the network at frequency f[n]
    
    l = z.size
    s = np.zeros((l,2,2),dtype=np.complex_)
    s[:,0,0] = np.ones(l)
    s[:,0,1] = z   
    s[:,1,0] = np.zeros(l)
    s[:,1,1] = np.ones(l)
    return s
 
def admittance (y):
    
    #for 2-port network with shunt admittance y (y is array if frequency dependent)
    #returns array s of size l*2*2 where s[n] = is the 2*2 paramater matrix
    #for the network at frequency f[n]
    
    l = y.size
    s = np.zeros((l,2,2),dtype=np.complex_)
    s[:,0,0] = np.ones(l)
    s[:,0,1] = np.zeros(l)
    s[:,1,0] = y
    s[:,1,1] = np.ones(l)
    return s

def shunt_cap(c,w):
    
    #for 2-port network with shunt capacitance c specified at complex frequencies w [rad/s]
    #returns array s of size l*2*2 where s[n] = is the 2*2 paramater matrix
    #for the network at frequency f[n]
    
    l = w.size
    s = np.zeros((l,2,2),dtype=np.complex_)
    s[:,0,0] = np.ones(l)
    s[:,0,1] = np.zeros(l)  
    s[:,1,0] = (c*w)
    s[:,1,1] = np.ones(l)
    return s


def series_cap(c,w):
    
    #for 2-port network with series capacitance c specified at complex frequencies w [rad/s]
    #returns array s of size l*2*2 where s[n] = is the 2*2 paramater matrix
    #for the network at frequency f[n]

    l = c.size
    s = np.zeros((l,2,2),dtype=np.complex_)
    s[:,0,0] = np.ones(l)
    s[:,0,1] = 1/(c*w)
    s[:,1,0] = np.zeros(l)
    s[:,1,1] = np.ones(l)
    return s

def series(networks):
    #Series combination of 2-port networks. Networks is a numpy array 
    # containing the A,B,C,D matrix parameters of networks to be combined 
    # in series

    out = np.matmul(networks[0],networks[1])
    
    for i in range (networks.shape[0]-2):
        out = np.matmul(out,networks[i+2])
        
    return out


def freq2impulse(H, f):
    #Returns the impulse response, h, and (optionally) the step response,
    #hstep, for a system with complex frequency response stored in the array H
    #and corresponding frequency vector f.  The time array is
    #returned in t.  The frequency array must be linearly spaced.

        Hd = np.concatenate((H,np.conj(np.flip(H[1:H.size-1]))))
        h = np.real(np.fft.ifft(Hd))
        #hstep = sp.convolve(h,np.ones(h.size))
        #hstep = hstep[0:h.size]
        t= np.linspace(0,1/f[1],h.size+1)
        t = t[0:-1]
        
        return h,t
        
def sparam(s11,s12,s21,s22,z0,f):
    
    #ABCD matrix description of a 2-port network with S-parameters
    #specified at the frequencies f in row vectors s11,s12,s21,s22
    
    #f should be a row vector
    
    #z0 is the characteristic impedance used for the S-parameter
    #measurements
    
    #Returns a structure containing the 2-port A,B,C,D matrix entries
    #at the frequencies in f: s.A, s.B, s.C, s.D
    
    s = np.zeros((f.size,2,2),dtype=np.complex_)
    
    s[:,0,0] = (1 + s11 - s22 - (s11*s22 - s12*s21))/(2*s21);
    s[:,0,1] = (z0*(1+s11 + s22 + (s11*s22 - s12*s21)))/(2*s21);
    s[:,1,0] = (1 - s11 - s22 + (s11*s22 - s12*s21))/(2*z0*s21);
    s[:,1,1] = (1 - s11 + s22 - (s11*s22 - s12*s21))/(2*s21);

    return s    


def zero_pad(H, f, t_d):
    '''Pads discrete time transfer function with zeros in to meet desired timestep in time domain
    
    Parameters
    ----------
    H: array
        Discrete time transfer function
    
    f: array
        frequency vector
    
    t_d : float 
        desired timestep
    
    
    Returns
    -------
    H_zero_pad: array
        zero-padded transfer function
    
    f_zero_pad: array
        extended frequency vector to match H_zero_pad
    
    h_0: array
        impulse response of zero-padded TF
    
    t_0: array
        time vector corresponding to h_0
    '''
    
    #frequency step
    f_step = f[1]

    #max frequency
    f_max = f[-1]

    #Desired max frequency to get t_d after IDFT
    f_max_d = int(1/(2*t_d))

    #extend frequency vector to f_max_d
    f_zero_pad = np.hstack( (f,np.linspace(f_max,f_max_d,int((f_max_d-f_max)/f_step))))

    #pad TF with zeros
    H_zero_pad = np.hstack((H, np.zeros((f_zero_pad.size-H.size))))
    
    #Calculate impulse response of zero-padded TF
    h_0,t_0 = freq2impulse(H_zero_pad,f_zero_pad)
    
    return H_zero_pad, f_zero_pad, h_0, t_0