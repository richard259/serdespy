"""Functions for time-domain signal processing and reciever equalization

"""

from serdespy.chmodel import *
import numpy as np
import skrf as rf
import scipy as sp

class Signal:
    """Class to represent time domain signal at reciever


    """
    
    def __init__(self, signal, steps_per_symbol, t_step, voltage_levels, crop = True):
        #self.signal_org = np.copy(signal)
        self.steps_per_symbol = steps_per_symbol
        self.t_step = t_step
        self.voltage_levels = voltage_levels
        
        self.t_symbol = self.steps_per_symbol*self.t_step
        self.baud_rate = 1/self.t_symbol
        self.frequency = 1/(2*self.t_symbol)
        
        if (crop):
        #shift signal so that every index i*steps_per_symbol is the index at wich to slice the signal
            self.signal_org = slicer(np.copy(signal), steps_per_symbol)
        
        else:
            self.signal_org = np.copy(signal)
        
        
        self.signal = np.copy(self.signal_org)
        
    def noise(self, stdev):
        self.signal = np.copy(self.signal_org) + np.random.normal(scale=stdev, size = self.signal_org.size)
        
    def DFE(self, tap_weights,threshold):
        
        signal_out =  np.copy(self.signal)
        n_taps = tap_weights.size
        n_symbols = int(round(self.signal.size/self.steps_per_symbol))
        half_symbol = int(round(self.steps_per_symbol/2))
        taps = np.zeros(n_taps)
        
        for symbol_idx in range(n_symbols-1):
            
            idx = symbol_idx*self.steps_per_symbol
            
            #decide on value of current bit and update taps
            if signal_out[idx] < threshold:
                taps = np.hstack((self.voltage_levels[0], taps[:-1]))
            else:
                taps = np.hstack((self.voltage_levels[1], taps[:-1]))
            
            #apply feedback to signal
            feedback = np.sum(taps*tap_weights)

            signal_out[idx+half_symbol:idx+self.steps_per_symbol+half_symbol] -= feedback

            
        self.signal = signal_out
        
    def dfe_with_precursor(self, tap_weights,n_taps_pre,threshold):
        
        signal_out =  np.copy(self.signal)
        
        n_symbols = int(round(self.signal.size/self.steps_per_symbol))
        half_symbol = int(round(self.steps_per_symbol/2))
                
        taps = np.zeros(tap_weights.size)
        
        for symbol_idx in range(n_taps_pre,n_symbols-n_taps_pre):
            
            idx_main = symbol_idx*self.steps_per_symbol
            idx_pre = (symbol_idx-n_taps_pre)*self.steps_per_symbol
            
            #decide on value of current bit and update taps
            if signal_out[idx_main] < threshold:
                taps = np.hstack((self.voltage_levels[0], taps[:-1]))
            else:
                taps = np.hstack((self.voltage_levels[1], taps[:-1]))
            
            #apply feedback to signal
            feedback = np.sum(taps*tap_weights)

            signal_out[idx_pre-half_symbol:idx_pre+self.steps_per_symbol-half_symbol] -= feedback

            
        self.signal_eq = signal_out
        
    def FFE(self,tap_weights, n_taps_pre):
        
        #equalized signal
        signal_out =  np.copy(self.signal)
        
        n_taps = tap_weights.size
                
        taps = np.zeros(n_taps)
        
        #create vector that has offset from main cursor for each tap
        taps_offset = np.zeros(n_taps)
        for i in range(n_taps):
            taps_offset[i] = (i-n_taps_pre)*self.steps_per_symbol
        
       # taps_offset = int(taps_offset)
        taps_offset = taps_offset.astype(int)
        
        #print(taps_offset)
        
        #apply FFE to each element in signal vector
        for idx in range(signal_out.size):
            
            #fill in taps
            for tap in range(tap_weights.size):
                
                #if at beginning or end of signal, pad taps with 0s
                if (idx+taps_offset[tap]<0) or ((idx+taps_offset[tap])>=self.signal.size):
                    taps[tap] = 0
                    
                else:
                
                #fill in tap values
                #print(tap,idx,taps_offset[tap])
                
                    taps[tap] = signal_out[idx+taps_offset[tap]]

            #calculate adjustment to current idx
            adjust  = np.sum(taps*tap_weights)
            
            #apply adjustment to signal
            signal_out[idx] += adjust
            
        self.signal = signal_out
            
        
def nrz_input(steps_per_symbol, data_in, voltage_levels):
    
    """Genterates NRZ (PAM-2) transmitter waveform from binary sequence

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

    """
    
    signal = np.zeros(steps_per_symbol*data_in.size)
    
    for i in range(data_in.size):
        if i==0:
            if (data_in[0]==0):
                signal[i*steps_per_symbol:(i+1)*steps_per_symbol] = np.ones(steps_per_symbol)*voltage_levels[0]
            else:
                signal[i*steps_per_symbol:(i+1)*steps_per_symbol]  = np.ones(steps_per_symbol)*voltage_levels[1]
        else:
            if (data_in[i]==0):
                signal[i*steps_per_symbol:(i+1)*steps_per_symbol] = np.ones(steps_per_symbol)*voltage_levels[0]
            else:
                signal[i*steps_per_symbol:(i+1)*steps_per_symbol] = np.ones(steps_per_symbol)*voltage_levels[1]
        if (i%100000 == 0):
            print('i=',i)
    
    return signal

#TODO: comment functions below

def slicer(signal, steps_per_symbol):
    
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
    
    data = np.zeros(int(signal.size/steps_per_symbol), dtype = np.uint8)
    
    for i in range(data.size):
        if signal[i*steps_per_symbol] > threshold:
            data[i] = 1
        else:
            data[i] = 0
            
    return data
