
from .chmodel import *
import numpy as np
import skrf as rf
import scipy as sp
import matplotlib.pyplot as plt
            
def pam4_decision(x,voltage_levels):
    l = (voltage_levels[0]+voltage_levels[1])/2
    m = (voltage_levels[1]+voltage_levels[2])/2
    h = (voltage_levels[2]+voltage_levels[3])/2
    
    if x<l:
        return 0
    elif x<m:
        return 1
    elif x<h:
        return 2
    else:
        return 3

def nrz_decision(x,voltage_levels):
    threshold = (voltage_levels[0]+voltage_levels[1])/2
    
    if x<threshold:
        return 0
    else:
        return 1


def nrz_input(samples_per_symbol, data_in, voltage_levels):
    
    """Genterates  ideal, square, NRZ (PAM-2) transmitter waveform from binary sequence

    Parameters
    ----------
    samples_per_symbol: int
        timesteps per bit
    
    length: int
        length of desired time-domain signal
    
    data_in: array
        binary sequence to input, must be longer than than length/samples_per_symbol
    
    voltage levels: array
        definition of voltages corresponding to 0 and 1. 
        voltage_levels[0] = voltage corresponding to 0 bit, 
        voltage_levels[1] = voltage corresponding to 1 bit
    
    length: float
        timestep of time domain signal
    
    Returns
    -------
    signal: array
        square waveform at trasmitter corresponding to data_in

    """
    
    signal = np.zeros(samples_per_symbol*data_in.size)
    
    for i in range(data_in.size):
        if (data_in[i] == 0):
            signal[i*samples_per_symbol:(i+1)*samples_per_symbol] = np.ones(samples_per_symbol)*voltage_levels[0]
        elif(data_in[i] == 1):
            signal[i*samples_per_symbol:(i+1)*samples_per_symbol]  = np.ones(samples_per_symbol)*voltage_levels[1]
        else:
            print('unexpected symbol in data_in')
            return False
            
        #if (i%100000 == 0):
         #   print('i=',i)
    
    return signal

def pam4_input_BR(data_in, voltage_levels):
      
    signal = np.zeros(data_in.size)
    
    for i in range(data_in.size):
        if (data_in[i]==0):
            signal[i] = voltage_levels[0]
        elif (data_in[i]==1):
            signal[i] = voltage_levels[1]
        elif (data_in[i]==2):
            signal[i] = voltage_levels[2]
        elif (data_in[i]==3):
            signal[i] = voltage_levels[3]
        else:
            print('unexpected symbol in data_in')
            return False
    
    return signal




def pam4_input(samples_per_symbol, data_in, voltage_levels):
    
    """Genterates ideal, square, PAM-4 transmitter waveform from binary sequence

    Parameters
    ----------
    samples_per_symbol: int
        timesteps per bit
    
    length: int
        length of desired time-domain signal
    
    data_in: array
        quaternary sequence to input, must be longer than than length/samples_per_symbol
    
    voltage levels: array
        definition of voltages corresponding to symbols. 
        voltage_levels[0] = voltage corresponding to 0 symbol, 
        voltage_levels[1] = voltage corresponding to 1 symbol
        voltage_levels[2] = voltage corresponding to 2 symbol
        voltage_levels[3] = voltage corresponding to 3 symbol
    
    length: float
        timestep of time domain signal
    
    Returns
    -------
    signal: array
        square waveform at trasmitter corresponding to data_in

    """
    
    signal = np.zeros(samples_per_symbol*data_in.size)
    
    for i in range(data_in.size):
        if (data_in[i]==0):
            signal[i*samples_per_symbol:(i+1)*samples_per_symbol] = np.ones(samples_per_symbol)*voltage_levels[0]
        elif (data_in[i]==1):
            signal[i*samples_per_symbol:(i+1)*samples_per_symbol] = np.ones(samples_per_symbol)*voltage_levels[1]
        elif (data_in[i]==2):
            signal[i*samples_per_symbol:(i+1)*samples_per_symbol] = np.ones(samples_per_symbol)*voltage_levels[2]
        elif (data_in[i]==3):
            signal[i*samples_per_symbol:(i+1)*samples_per_symbol] = np.ones(samples_per_symbol)*voltage_levels[3]
        else:
            print('unexpected symbol in data_in')
            return False

        if (i%100000 == 0):
            print('i=',i)
    
    return signal

#TODO: optimize for BR signal

def shift_signal(signal, samples_per_symbol):
    """Shifts signal vector so that 0th element is at centre of eye, heuristic

    Parameters
    ----------
    signal: array
        signal vector at RX
        
    samples_per_symbol: int
        number of timesteps per one bit
    
    Returns
    -------
    array
        signal shifted so that 0th element is at centre of eye

    """
    
    #Loss function evaluated at each step from 0 to steps_per_signal
    loss_vec = np.zeros(samples_per_symbol)
    
    for i in range(samples_per_symbol):
        
        samples = signal[i::samples_per_symbol]
        
        #add loss for samples that are close to threshold voltage
        loss_vec[i] += np.sum(abs(samples)**2)
        
    #find shift index with least loss
    shift = np.where(loss_vec == np.max(loss_vec))[0][0]
    
   # plt.plot(np.linspace(0,samples_per_symbol-1,samples_per_symbol),loss_vec)
   # plt.show()
    
    return np.copy(signal[shift+1:])


def nrz_a2d(signal, samples_per_symbol, threshold):
    """slices signal and compares to threshold to convert analog signal vector to binary sequence
        signal is sampled at indeicies that are multiples of samples_per_symbol

    Parameters
    ----------
    signal: array
        signal vector at RX
        
    samples_per_symbol: int
        number of timesteps per one bit
        
    threshold: float
        threshold voltage to decide if bit is 1 or 0
    
    Returns
    -------
    data : array
        binary sequence generated from signal

    """
    data = np.zeros(int(signal.size/samples_per_symbol), dtype = np.uint8)
    
    for i in range(data.size):
        if signal[i*samples_per_symbol] > threshold:
            data[i] = 1
        else:
            data[i] = 0
            
    return data

def pam4_a2d(signal, samples_per_symbol, voltage_levels):
    """slices signal and compares to voltage_levelsto convert analog signal vector to binary sequence
        signal is sampled at indeicies that are multiples of samples_per_symbol

    Parameters
    ----------
    signal: array
        signal vector at RX
        
    samples_per_symbol: int
        number of timesteps per one bit
        
    voltage_levels: array
    
    Returns
    -------
    data : array
        binary sequence generated from signal

    """
    data = np.zeros(int(signal.size/samples_per_symbol), dtype = np.uint8)
    
    for i in range(data.size):
        data[i] = pam4_decision(signal[i*samples_per_symbol],voltage_levels)
            
    return data


def channel_coefficients(pulse_response, t, samples_per_symbol, n_precursors, n_postcursors):

    n_cursors = n_precursors + n_postcursors + 1
    channel_coefficients = np.zeros(n_cursors)
    
    t_vec = np.zeros(n_cursors)
    xcoords = []
    half_symbol = int(round(samples_per_symbol/2))
    
    #find peak of pulse response
    max_idx = np.where(pulse_response == np.amax(pulse_response))[0][0]
    
    
    for cursor in range(n_cursors):
        
        a = cursor - n_precursors
        
        channel_coefficients[cursor] = pulse_response[max_idx+a*samples_per_symbol]
        
        #for plotting
        xcoords = xcoords + [1e9*t[max_idx+a*samples_per_symbol-half_symbol]]
        t_vec[a+n_precursors] = t[max_idx + a*samples_per_symbol]
    xcoords = xcoords + [1e9*t[max_idx+(n_postcursors+1)*samples_per_symbol-half_symbol]]
    
    
    #plot pulse response and cursor samples
    plt.figure()
    plt.plot(t_vec*1e9, channel_coefficients, 'o',label = 'Cursor samples')
    plt.plot(t*1e9,pulse_response, label = 'Pulse Response')
    plt.xlabel("Time [ns]")
    plt.ylabel("[V]")
    
    ll = t[max_idx-samples_per_symbol*(n_precursors+2)]*1e9
    ul = t[max_idx+samples_per_symbol*(n_postcursors+2)]*1e9
    
    #print(ll,ul)
    plt.xlim([ll,ul])
    plt.title("Channel Coefficients")
    plt.legend()
    for xc in xcoords:
        plt.axvline(x=xc,color = 'grey',label ='UIs')
    
    return channel_coefficients
