"""Functions for time and frequency domain resampling

"""

from serdespy.chmodel import *
import numpy as np
import skrf as rf
import samplerate

#ratio = input timestep/output_timestep
#samplerate.resample(input_signal, ratio, 'sinc_best')
    

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