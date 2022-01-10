from .signal import *
from .prs import *
from .eye_diagram import *
import numpy as np
import skrf as rf

def tx_jitter(UI,samples_per_symbol,ideal):
    """Generates the TX output waveform including jitter

    Parameters
    ----------
    UI : float
        unit interval time
    
    samples_per_symbol : int
        number of samples per symbol
        
    ideal: list
        output of nrz_input or pam4_input, provides the ideal waveform on which to add TX jitter

    Returns
    -------
    superposition : list
        TX output waveform including jitter
        
    non_ideal :  list
        list of Gaussian distributed epsilon (jitter) values
    """

    #generate random Gaussian distributed TX jitter values
    epsilon = np.random.normal(0,0.025*UI,len(ideal)//samples_per_symbol)
    epsilon.clip(UI)
    epsilon[0]=0

    #calculate time duration of each sample
    sample_time = UI/samples_per_symbol

    #initializes non_ideal (jitter) array
    non_ideal = np.zeros_like(ideal)

    #populates non_ideal array to create TX jitter waveform
    for symbol_index,symbol_epsilon in enumerate(epsilon):
        epsilon_duration = int(round(symbol_epsilon/sample_time))
        start = int(symbol_index*samples_per_symbol)
        end = int(start+epsilon_duration)
        flip=1
        if symbol_index==0:
            continue
        if symbol_epsilon<0:
            start,end=end,start
            flip=-1
        non_ideal[start:end]=flip*(ideal[symbol_index*samples_per_symbol-samples_per_symbol]-ideal[symbol_index*samples_per_symbol])
    
    #calculate TX output waveform
    superposition = ideal+non_ideal

    return superposition, non_ideal