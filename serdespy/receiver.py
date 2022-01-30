from .signal import *
from .chmodel import *
import numpy as np
import skrf as rf
import scipy as sp
import matplotlib.pyplot as plt


class Receiver:
    """Class to represent time domain signal at receiver


    """
    
    def __init__(self, signal, samples_per_symbol, t_step, voltage_levels, shift = True, main_cursor = 1):
        #self.signal_org = np.copy(signal)
        self.samples_per_symbol = samples_per_symbol
        self.t_step = t_step
        self.voltage_levels = voltage_levels
        
        self.t_symbol = self.samples_per_symbol*self.t_step
        self.baud_rate = 1/self.t_symbol
        self.frequency = 1/(2*self.t_symbol)
        
        self.main_cursor = main_cursor
        
        if (shift):
        #shift signal so that every index i*samples_per_symbol is the index at wich to slice the signal
            self.signal_org = shift_signal(np.copy(signal), samples_per_symbol)
        
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
        n_symbols = int(round(self.signal.size/self.samples_per_symbol))
        half_symbol = int(round(self.samples_per_symbol/2))
        taps = np.zeros(n_taps)
        
        for symbol_idx in range(n_symbols-1):
            
            idx = symbol_idx*self.samples_per_symbol
            
            #decide on value of current bit
            symbol = nrz_decision(signal_out[idx],self.voltage_levels)
            
            #update_taps
            taps = np.hstack((self.voltage_levels[symbol], taps[:-1]))
            
            #apply feedback to signal
            feedback = np.sum(taps*tap_weights)

            signal_out[idx+half_symbol:idx+self.samples_per_symbol+half_symbol] -= feedback

            
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
        n_symbols = int(round(self.signal.size/self.samples_per_symbol))
        half_symbol = int(round(self.samples_per_symbol/2))
        taps = np.zeros(n_taps)
        
        for symbol_idx in range(n_symbols-1):
            if (symbol_idx%10000 == 0):
                print('i=',symbol_idx)
            idx = symbol_idx*self.samples_per_symbol
            
            #decide on value of current bit 
            symbol = pam4_decision(signal_out[idx],self.voltage_levels)
            
            #update taps
            taps = np.hstack((self.voltage_levels[symbol], taps[:-1]))
            
            #apply feedback to signal
            feedback = np.sum(taps*tap_weights)

            signal_out[idx+half_symbol:idx+self.samples_per_symbol+half_symbol] -= feedback

        self.signal = signal_out
        
        
    def pam4_DFE_BR(self, tap_weights):
        
        signal_out =  np.copy(self.signal)
        n_taps = tap_weights.size
        n_symbols = int(round(self.signal.size/self.samples_per_symbol))
        #half_symbol = int(round(self.samples_per_symbol/2))
        taps = np.zeros(n_taps)
        
        for symbol_idx in range(n_symbols-1):
            #if (symbol_idx%10000 == 0):
            #    print('i=',symbol_idx)
                
            #idx = symbol_idx*self.samples_per_symbol
            
            #decide on value of current bit 
            symbol = pam4_decision(signal_out[symbol_idx],self.voltage_levels*self.main_cursor)
            
            #update taps
            taps = np.hstack((self.voltage_levels[symbol], taps[:-1]))
            
            #apply feedback to signal
            feedback = np.sum(taps*tap_weights)

            signal_out[symbol_idx+1] -= feedback

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
                
        tap_filter = np.zeros((n_taps-1)*self.samples_per_symbol+1)
        
        
        
        for i in range(n_taps):
            tap_filter[i*self.samples_per_symbol] = tap_weights[i]
            
        
        length = self.signal.size
        self.signal = np.convolve(self.signal,tap_filter)
        #shift = round((n_taps_pre-n_taps)*self.samples_per_symbol)
        self.signal = self.signal[n_taps_pre*self.samples_per_symbol:n_taps_pre*self.samples_per_symbol+length]
        
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
    