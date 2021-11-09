"""Functions for plotting eye diagram

"""
import numpy as np
import matplotlib.pyplot as plt

#TODO: make better eye diagram plotter 

def simple_eye(signal, window_len, ntraces, tstep, title):
    """Genterates PRBS31 sequence

    Parameters
    ----------
    signal: array
        signal to be plotted
    
    window_len: int
        number of time steps in eye diagram x axis
    
    ntraces: int
        number of traces to be plotted
    
    tstep: float
        timestep of time domain signal
    
    title: 
        title of the plot
    """

    signal_crop = signal[0:ntraces*window_len]
    traces = np.split(signal_crop,ntraces)

    t = np.linspace(-(tstep*(window_len-1))/2,(tstep*(window_len-1))/2, window_len)
    
    plt.figure()
    for i in range(ntraces):
        plt.plot(t*1e12,np.reshape((traces[i][:]),window_len))
        plt.title(title)
        plt.xlabel('[ps]')
        plt.ylabel('[V]')
    

    