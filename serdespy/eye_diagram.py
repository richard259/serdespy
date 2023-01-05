"""Functions for plotting eye diagrams

"""
import numpy as np
import matplotlib.pyplot as plt
from .signal import *

def simple_eye(signal, window_len, ntraces, tstep, title, res=600, linewidth=0.15):
    """Genterates simple eye diagram

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
        
    res: int, optional
        DPI resolution of the figure
        
    linewidth: float, optional
        width of lines in figure
    """

    signal_crop = signal[0:ntraces*window_len]
    traces = np.split(signal_crop,ntraces)

    t = np.linspace(-(tstep*(window_len-1))/2,(tstep*(window_len-1))/2, window_len)
    
    plt.figure(dpi=res)
    for i in range(ntraces):
        plt.plot(t*1e12,np.reshape((traces[i][:]),window_len), color = 'blue', linewidth = linewidth)
        plt.title(title)
        plt.xlabel('[ps]')
        plt.ylabel('[V]')
    
    return True


def rx_jitter_eye(signal, window_len, ntraces, n_symbols, tstep, title,  stdev, res=600, linewidth=0.15,):
    """Genterates eye diagram with jitter introduved by splitting traces and applying 
    horizontal shift with gaussian distribution

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
        
    stdev: float
        RMS value of gaussian jitter
    
    res: int, optional
        DPI resolution of the figure
        
    linewidth: float, optional
        width of lines in figure
    """
    epsilon = np.random.normal(0,stdev,n_symbols)
    
    epsilon.clip(window_len*tstep)
    epsilon[0]=0
    
    signal_crop = signal[0:ntraces*window_len]
    traces = np.split(signal_crop,ntraces)
    
    plt.figure(dpi=res)
    for symbol_index,symbol_epsilon in enumerate(epsilon):
        epsilon_duration = int(round(symbol_epsilon/tstep))
        t = np.linspace( -tstep * (((window_len-1))/2 + epsilon_duration ) ,tstep * (((window_len-1))/2 + epsilon_duration ), window_len)
        plt.plot(t*1e12,np.reshape((traces[symbol_index][:]),window_len), color = 'blue', linewidth = linewidth)
    
        
    plt.title(title)
    plt.xlabel('[ps]')
    plt.ylabel('[V]')
    plt.xlim([-tstep*1e12*(window_len)/2, tstep*1e12* (window_len)/2])
    
    return True