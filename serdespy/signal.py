"""Functions for time-domain signal processing and reciever equalization

"""

from .chmodel import *
import numpy as np
import skrf as rf
import scipy as sp
import matplotlib.pyplot as plt

class Receiver:
    """Class to represent time domain signal at reciever


    """
    
    def __init__(self, signal, steps_per_symbol, t_step, voltage_levels, shift = True):
        #self.signal_org = np.copy(signal)
        self.steps_per_symbol = steps_per_symbol
        self.t_step = t_step
        self.voltage_levels = voltage_levels
        
        self.t_symbol = self.steps_per_symbol*self.t_step
        self.baud_rate = 1/self.t_symbol
        self.frequency = 1/(2*self.t_symbol)
        
        if (shift):
        #shift signal so that every index i*steps_per_symbol is the index at wich to slice the signal
            self.signal_org = shift_signal(np.copy(signal), steps_per_symbol)
        
        else:
            self.signal_org = np.copy(signal)
        
        
        self.signal = np.copy(self.signal_org)
    
    def reset(self):
        """Resets Signal to original (unequalized, no noise) signal"""
    
        self.signal = np.copy(self.signal_org)
    
    def noise(self, stdev):
        """Adds 0-mean gaussian noise to signal
    
        Parameters
        ----------
        stdev : float
            standard deviation of noise
        """
    
        self.signal = np.copy(self.signal_org) + np.random.normal(scale=stdev, size = self.signal_org.size)
        
    def nrz_DFE(self, tap_weights):
        """Behavioural model of DFE for NRZ signal. Input signal is self.signal, this method modifies self.signal
    
        Parameters
        ----------
        tap_weights: array
            DFE tap weights
        """
        
        signal_out =  np.copy(self.signal)
        n_taps = tap_weights.size
        n_symbols = int(round(self.signal.size/self.steps_per_symbol))
        half_symbol = int(round(self.steps_per_symbol/2))
        taps = np.zeros(n_taps)
        
        for symbol_idx in range(n_symbols-1):
            
            idx = symbol_idx*self.steps_per_symbol
            
            #decide on value of current bit
            symbol = nrz_decision(signal_out[idx],self.voltage_levels)
            
            #update_taps
            taps = np.hstack((self.voltage_levels[symbol], taps[:-1]))
            
            #apply feedback to signal
            feedback = np.sum(taps*tap_weights)

            signal_out[idx+half_symbol:idx+self.steps_per_symbol+half_symbol] -= feedback

            
        self.signal = signal_out
        
    def pam4_DFE(self, tap_weights):
        """Behavioural model of DFE for PAM-4 signal. Input signal is self.signal, this method modifies self.signal.
    
        Parameters
        ----------
        tap_weights: array
            DFE tap weights
        """
        
        signal_out =  np.copy(self.signal)
        n_taps = tap_weights.size
        n_symbols = int(round(self.signal.size/self.steps_per_symbol))
        half_symbol = int(round(self.steps_per_symbol/2))
        taps = np.zeros(n_taps)
        
        for symbol_idx in range(n_symbols-1):
            if (symbol_idx%10000 == 0):
                print('i=',symbol_idx)
            idx = symbol_idx*self.steps_per_symbol
            
            #decide on value of current bit 
            symbol = pam4_decision(signal_out[idx],self.voltage_levels)
            
            #update taps
            taps = np.hstack((self.voltage_levels[symbol], taps[:-1]))
            
            #apply feedback to signal
            feedback = np.sum(taps*tap_weights)

            signal_out[idx+half_symbol:idx+self.steps_per_symbol+half_symbol] -= feedback

        self.signal = signal_out
        
        
    def pam4_DFE_BR(self, tap_weights):
        
        signal_out =  np.copy(self.signal)
        n_taps = tap_weights.size
        n_symbols = int(round(self.signal.size/self.steps_per_symbol))
        #half_symbol = int(round(self.steps_per_symbol/2))
        taps = np.zeros(n_taps)
        
        for symbol_idx in range(n_symbols-1):
            if (symbol_idx%10000 == 0):
                print('i=',symbol_idx)
            idx = symbol_idx*self.steps_per_symbol
            
            #decide on value of current bit 
            symbol = pam4_decision(signal_out[idx],self.voltage_levels)
            
            #update taps
            taps = np.hstack((self.voltage_levels[symbol], taps[:-1]))
            
            #apply feedback to signal
            feedback = np.sum(taps*tap_weights)

            signal_out[idx+1] -= feedback

        self.signal = signal_out
        
    def FFE(self,tap_weights, n_taps_pre):
        """Behavioural model of FFE. Input signal is self.signal, this method modifies self.signal
    
        Parameters
        ----------
        tap_weights: array
            DFE tap weights
            
        n_taps_pre: int
            number of precursor taps
        """
        
        n_taps = tap_weights.size
                
        tap_filter = np.zeros((n_taps-1)*self.steps_per_symbol+1)
        
        
        
        for i in range(n_taps):
            tap_filter[i*self.steps_per_symbol] = tap_weights[i]
            
        #print(tap_filter)
        
        length = self.signal.size
        self.signal = np.convolve(self.signal,tap_filter)
        #shift = round((n_taps_pre-n_taps)*self.steps_per_symbol)
        self.signal = self.signal[n_taps_pre*self.steps_per_symbol:n_taps_pre*self.steps_per_symbol+length]
        
    def CTLE(self, b,a,f):
                
        """Behavioural model of continuous-time linear equalizer (CTLE). Input signal is self.signal, this method modifies self.signal
    
        Parameters
        ----------
        
        b: array
            coefficients in numerator of ctle transfer function
        a: array
            coefficients in denomenator of ctle transfer function
    
        """
            
        #create freqency vector in rad/s
        w = f/(2*np.pi)
        
        #compute Frequency response of CTLE at frequencies in w vector
        w, H_ctle = sp.signal.freqs(b, a, w)
        
        #convert to impulse response
        h_ctle, t_ctle = freq2impulse(H_ctle,f)
        
        #check that time_steps match
        if ((t_ctle[1]-self.t_step)/self.t_step>1e-9):
            print("Invalid f vector, need length(f)/f[1] = ", self.t_step)
            return False
        
        #convolve signal with impulse response of CTLE
        signal_out = sp.signal.fftconvolve(self.signal, h_ctle[:100], mode = 'same')
        
        self.signal = np.copy(signal_out)
    
            
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


def nrz_input(steps_per_symbol, data_in, voltage_levels):
    
    """Genterates  ideal, square, NRZ (PAM-2) transmitter waveform from binary sequence

    Parameters
    ----------
    steps_per_symbol: int
        timesteps per bit
    
    length: int
        length of desired time-domain signal
    
    data_in: array
        binary sequence to input, must be longer than than length/steps_per_symbol
    
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
    
    signal = np.zeros(steps_per_symbol*data_in.size)
    
    for i in range(data_in.size):
        if (data_in[i] == 0):
            signal[i*steps_per_symbol:(i+1)*steps_per_symbol] = np.ones(steps_per_symbol)*voltage_levels[0]
        elif(data_in[i] == 1):
            signal[i*steps_per_symbol:(i+1)*steps_per_symbol]  = np.ones(steps_per_symbol)*voltage_levels[1]
        else:
            print('unexpected symbol in data_in')
            return False
            
        #if (i%100000 == 0):
         #   print('i=',i)
    
    return signal

def pam4_input(steps_per_symbol, data_in, voltage_levels):
    
    """Genterates ideal, square, PAM-4 transmitter waveform from binary sequence

    Parameters
    ----------
    steps_per_symbol: int
        timesteps per bit
    
    length: int
        length of desired time-domain signal
    
    data_in: array
        binary sequence to input, must be longer than than length/steps_per_symbol
    
    voltage levels: array
        definition of voltages corresponding to 0 and 1. 
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
    
    signal = np.zeros(steps_per_symbol*data_in.size)
    
    for i in range(data_in.size):
        if (data_in[i]==0):
            signal[i*steps_per_symbol:(i+1)*steps_per_symbol] = np.ones(steps_per_symbol)*voltage_levels[0]
        elif (data_in[i]==1):
            signal[i*steps_per_symbol:(i+1)*steps_per_symbol] = np.ones(steps_per_symbol)*voltage_levels[1]
        elif (data_in[i]==2):
            signal[i*steps_per_symbol:(i+1)*steps_per_symbol] = np.ones(steps_per_symbol)*voltage_levels[2]
        elif (data_in[i]==3):
            signal[i*steps_per_symbol:(i+1)*steps_per_symbol] = np.ones(steps_per_symbol)*voltage_levels[3]
        else:
            print('unexpected symbol in data_in')
            return False

        if (i%100000 == 0):
            print('i=',i)
    
    return signal

#TODO: comment functions below

def shift_signal(signal, steps_per_symbol):
    """Shifts signal vector so that 0th element is at centre of eye, heuristic

    Parameters
    ----------
    signal: array
        signal vector at RX
        
    steps_per_symbol: int
        number of timesteps per one bit
    
    Returns
    -------
    array
        signal shifted so that 0th element is at centre of eye

    """
    
    #Loss function evaluated at each step from 0 to steps_per_signal
    loss_vec = np.zeros(steps_per_symbol)
    
    for i in range(steps_per_symbol):
        
        samples = signal[i::steps_per_symbol]
        
        #add loss for samples that are close to threshold voltage
        loss_vec[i] += np.sum(abs(samples)**2)
        
    #find shift index with least loss
    shift = np.where(loss_vec == np.max(loss_vec))[0][0]
    
   # plt.plot(np.linspace(0,steps_per_symbol-1,steps_per_symbol),loss_vec)
   # plt.show()
    
    return np.copy(signal[shift+1:])


def nrz_a2d(signal, steps_per_symbol, threshold):
    """slices signal and compares to threshold to convert analog signal vector to binary sequence
        signal is sampled at indeicies that are multiples of steps_per_symbol

    Parameters
    ----------
    signal: array
        signal vector at RX
        
    steps_per_symbol: int
        number of timesteps per one bit
        
    threshold: float
        threshold voltage to decide if bit is 1 or 0
    
    Returns
    -------
    data : array
        binary sequence generated from signal

    """
    data = np.zeros(int(signal.size/steps_per_symbol), dtype = np.uint8)
    
    for i in range(data.size):
        if signal[i*steps_per_symbol] > threshold:
            data[i] = 1
        else:
            data[i] = 0
            
    return data

def pam4_a2d(signal, steps_per_symbol, voltage_levels):
    """slices signal and compares to voltage_levelsto convert analog signal vector to binary sequence
        signal is sampled at indeicies that are multiples of steps_per_symbol

    Parameters
    ----------
    signal: array
        signal vector at RX
        
    steps_per_symbol: int
        number of timesteps per one bit
        
    voltage_levels: array
    
    Returns
    -------
    data : array
        binary sequence generated from signal

    """
    data = np.zeros(int(signal.size/steps_per_symbol), dtype = np.uint8)
    
    for i in range(data.size):
        data[i] = pam4_decision(signal[i*steps_per_symbol],voltage_levels)
            
    return data


def channel_coefficients(pulse_response, t, steps_per_symbol, n_precursors, n_postcursors):

    n_cursors = n_precursors + n_postcursors + 1
    channel_coefficients = np.zeros(n_cursors)
    
    t_vec = np.zeros(n_cursors)
    xcoords = []
    half_symbol = int(round(steps_per_symbol/2))
    
    #find peak of pulse response
    max_idx = np.where(pulse_response == np.amax(pulse_response))[0][0]
    
    
    for cursor in range(n_cursors):
        
        a = cursor - n_precursors
        
        channel_coefficients[cursor] = pulse_response[max_idx+a*steps_per_symbol]
        
        #for plotting
        xcoords = xcoords + [1e9*t[max_idx+a*steps_per_symbol-half_symbol]]
        t_vec[a+n_precursors] = t[max_idx + a*steps_per_symbol]
    xcoords = xcoords + [1e9*t[max_idx+(n_postcursors+1)*steps_per_symbol-half_symbol]]
    
    
    #plot pulse response and cursor samples
    plt.figure()
    plt.plot(t_vec*1e9, channel_coefficients, 'o',label = 'Cursor samples')
    plt.plot(t*1e9,pulse_response, label = 'Pulse Response')
    plt.xlabel("Time [ns]")
    plt.ylabel("[V]")
    
    ll = t[max_idx-steps_per_symbol*(n_precursors+2)]*1e9
    ul = t[max_idx+steps_per_symbol*(n_postcursors+2)]*1e9
    
    #print(ll,ul)
    plt.xlim([ll,ul])
    plt.title("Channel Coefficients")
    plt.legend()
    for xc in xcoords:
        plt.axvline(x=xc,color = 'grey',label ='UIs')
    
    return channel_coefficients
