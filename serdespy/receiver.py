from .signal import *
from .chmodel import *
import numpy as np
import skrf as rf
import scipy as sp
import matplotlib.pyplot as plt


class Receiver:
    """Class to build model of time domain signal at receiver 
    
    """
    
    def __init__(self, signal, samples_per_symbol, f_nyquist , voltage_levels, shift = True, main_cursor = 1):
        """Initialize Receiver
    
        Parameters
        ----------
        signal : array
            voltage waveform at input to reciever (after CTLE)
        
        samples_per_symbol:
            number of samples per UI
        
        t_step:
            timestep of signal
        
        voltage_levels:
            
        shift : bool
            if True, shifts signal so centre of eye opening is at beginning of signal
        
        main_cursor : float
            peak of pulse response, used for determining voltage level corresponding to pam4 symbols
        """
        
        self.samples_per_symbol = samples_per_symbol
        
        self.voltage_levels = voltage_levels
        
        
        self.f_nyquist = f_nyquist
        
        self.main_cursor = main_cursor
        
        #signal_org maintains a copy of the original signal
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
    
        
    
    def slice_signal(self):
        
        signal_BR = np.zeros(int(np.floor((self.signal.size/self.samples_per_symbol))))
        
        for i in range(signal_BR.size):
            signal_BR[i] = self.signal[i*self.samples_per_symbol]
        
        self.signal_BR = signal_BR
    
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
        
    def FFE_BR(self, tap_weights, n_taps_pre):
        """Behavioural model of FFE. Input signal is self.signal, this method modifies self.signal
    
        Parameters
        ----------
        tap_weights: array
            DFE tap weights
            
        n_taps_pre: int
            number of precursor taps
        """
        
        self.signal_BR = sp.signal.fftconvolve(self.signal_BR,tap_weights, mode="same")
    
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
        
        t = self.main_cursor*((self.voltage_levels[0]+self.voltage_levels[1])/2)
        
        for symbol_idx in range(n_symbols-1):
            
            idx = symbol_idx*self.samples_per_symbol
            
            #decide on value of current bit
            symbol = nrz_decision(signal_out[idx],t)
            
            #update_taps
            taps = np.hstack((self.voltage_levels[symbol], taps[:-1]))
            
            #apply feedback to signal
            feedback = np.sum(taps*tap_weights)

            signal_out[idx+half_symbol:idx+self.samples_per_symbol+half_symbol] -= feedback

            
        self.signal = signal_out
        
    def nrz_DFE_BR(self, tap_weights):
        """Behavioural model of DFE for NRZ signal. Input signal is self.signal, this method modifies self.signal
    
        Parameters
        ----------
        tap_weights: array
            DFE tap weights
        """
        
        signal_out =  np.copy(self.signal_BR)
        symbols_out = np.zeros(self.signal_BR.size, dtype = np.uint8)
        n_taps = tap_weights.size
        taps = np.zeros(n_taps)
        
        t = self.main_cursor*((self.voltage_levels[0]+self.voltage_levels[1])/2)
        
        for symbol_idx in range(len(self.signal_BR)-1):
            
            #decide on value of current bit
            symbols_out[symbol_idx] = nrz_decision(signal_out[symbol_idx],t)
            
            #update taps            
            taps[1:] = taps[:-1]
            taps[0] = self.voltage_levels[symbols_out[symbol_idx]]
            
            #apply decision feedback to next bit
            for i in range(n_taps):
                signal_out[symbol_idx+1] -= taps[i]*tap_weights[i]

            
        self.signal_BR = signal_out
        
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
    
        l = self.main_cursor*((self.voltage_levels[0]+self.voltage_levels[1])/2)
        m = self.main_cursor*((self.voltage_levels[1]+self.voltage_levels[2])/2)
        h = self.main_cursor*((self.voltage_levels[2]+self.voltage_levels[3])/2)
    
        for symbol_idx in range(n_symbols-1):
            
            idx = symbol_idx*self.samples_per_symbol
            
            #decide on value of current bit 
            symbol = pam4_decision(signal_out[idx],l,m,h)
            
            #update taps            
            taps[1:] = taps[:-1]
            taps[0] = self.voltage_levels[symbol]
            
            #apply feedback to signal
            feedback = np.sum(taps*tap_weights)

            signal_out[idx+half_symbol:idx+self.samples_per_symbol+half_symbol] -= feedback

        self.signal = signal_out
        
    
    def pam4_DFE_BR(self, tap_weights):
        
        signal_out =  np.copy(self.signal_BR)
        
        symbols_out = np.zeros(self.signal_BR.size, dtype = np.uint8)
        
        n_taps = tap_weights.size
        
        taps = np.zeros(n_taps)
        
        l = self.main_cursor*((self.voltage_levels[0]+self.voltage_levels[1])/2)
        m = self.main_cursor*((self.voltage_levels[1]+self.voltage_levels[2])/2)
        h = self.main_cursor*((self.voltage_levels[2]+self.voltage_levels[3])/2)
        
        for symbol_idx in range(len(self.signal_BR)-1):
            
            #decide on value of current bit 
            symbols_out[symbol_idx] = pam4_decision(signal_out[symbol_idx],l,m,h)
            
            #update taps            
            taps[1:] = taps[:-1]
            taps[0] = self.voltage_levels[symbols_out[symbol_idx]]
            
            #apply decision feedback to next bit
            for i in range(n_taps):
                signal_out[symbol_idx+1] -= taps[i]*tap_weights[i]

        self.signal_BR = signal_out
        self.symbols_out = symbols_out
        

        
            