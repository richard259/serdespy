from .signal import *
from .chmodel import *
import numpy as np
import skrf as rf
import scipy as sp
import matplotlib.pyplot as plt
import samplerate

class Transmitter:
    """class to build model of time domain signal at transmitter 
    
    """
    
    def __init__(self, data, voltage_levels, frequency):
        """
        Initialize transmitter, stores data and converts to baud-rate-sampled voltage waveform
        
        Parameters
        ----------
        data : array
            Binary sequence containing {0,1} if NRZ
            Quaternary sequence containing {0,1,2,3} symbols if PAM-4
        
        voltage levels: array
            definition of voltages corresponding to symbols. 
        
        frequency: float
            2* symbol rate
        
        """
        
        #frequency and period
        self.f = frequency
        self.T = 1/self.f
        self.UI = self.T/2
        
        
        self.voltage_levels = voltage_levels
        self.data = data
        self.n_symbols = data.size
        
        
        #self.signal_FIR_BR = None
        self.FIR_enable = False
        
        #create ideal, baud-rate-sampled transmitter waveform
        if voltage_levels.size == 2:
            self.signal_BR = nrz_input_BR(data,voltage_levels)
        
        elif voltage_levels.size == 4:
            self.signal_BR = pam4_input_BR(data,voltage_levels)
        
        else:
            print ("Error: Voltage levels must have either size = 2 for NRZ signal or size = 4 for PAM4")
    
    def FIR(self, tap_weights):
        
        """Implements TX - FIR and creates  self.signal_FIR_BR = filtered, baud-rate sampled signal
        
        Parameters
        ----------
        
        tap_weights: array
            tap weights for tx fir
            last element should be 1, eg. for a 2-tap TX-FIR, with -0.1 and -0.2 coefficients, tap_weights = np.array([-0.1, -0.2, 1])
            
        """
        self.FIR_enable = True
        
        #do convolution to implement FIR
        self.signal_FIR_BR = sp.signal.fftconvolve(self.signal_BR,tap_weights, mode="same")

    def oversample(self, samples_per_symbol):
        """oversample baud-rate signal to create ideal, square transmitter waveform
        
        Parameters
        ----------
        
        samples_per_symbol:
            samples per UI of tx signal
            
        """
        
        
        self.samples_per_symbol = samples_per_symbol
        
        
        
        #TODO: use np.repeat
        
        
        #if we have FIR filtered data
        if self.FIR_enable:
            #oversampled = samplerate.resample(self.signal_FIR_BR,samples_per_symbol,converter_type='zero_order_hold')
            self.signal_ideal = np.repeat(self.signal_FIR_BR, samples_per_symbol)
        #if we are not using FIR
        else:
            #oversampled = samplerate.resample(self.signal_BR,samples_per_symbol,converter_type='zero_order_hold')
            self.signal_ideal = np.repeat(self.signal_BR, samples_per_symbol)
    
    def gaussian_jitter(self, stdev_div_UI = 0.025):
        """Generates the TX waveform from ideal, square, self.signal_ideal with jitter
    
        Parameters
        ----------
        stdev_div_UI : float
            standard deviation of jitter distribution as a pct of UI    
        """
    
        #generate random Gaussian distributed TX jitter values
        epsilon = np.random.normal(0,stdev_div_UI*self.UI,self.n_symbols)
        
        epsilon.clip(self.UI)
        epsilon[0]=0
    
        #calculate time duration of each sample
        sample_time = self.UI/self.samples_per_symbol
    
        #initializes non_ideal (jitter) array
        non_ideal = np.zeros_like(self.signal_ideal)
    
        #populates non_ideal array to create TX jitter waveform
        for symbol_index,symbol_epsilon in enumerate(epsilon):
            epsilon_duration = int(round(symbol_epsilon/sample_time))
            start = int(symbol_index*self.samples_per_symbol)
            end = int(start+epsilon_duration)
            flip=1
            if symbol_index==0:
                continue
            if symbol_epsilon<0:
                start,end=end,start
                flip=-1
            non_ideal[start:end]=flip*(self.signal_ideal[symbol_index*self.samples_per_symbol-self.samples_per_symbol]-self.signal_ideal[symbol_index*self.samples_per_symbol])
        
        #calculate TX output waveform
        self.signal = np.copy(self.signal_ideal+non_ideal)

    def tx_bandwidth(self, freq_bw = None, TF = None):
        """Passes TX signal through an LTI system to model non-ideal TX driver
        option to use custom transfer function, or use single-pole system with specified -3dB frequency 

        Parameters
        ----------
        freq_bw: float
            bandwidth frequency

        TF: list, optional
            TF[0] : numerator coefficients for tranfer function
            TF[1] : denominator coefficients for Transfer function
            
        """

        #Timestep
        dt = self.UI/self.samples_per_symbol

        #max frequency for constructing discrete transfer function
        max_f = 1/dt

        #max_f in rad/s
        max_w = max_f*2*np.pi

        #heuristic to get a reasonable impulse response length
        ir_length = int(4/(freq_bw*dt))

        #Calculate discrete transfer function
        if TF != None:
            w, H = sp.signal.freqs(TF[0], TF[1], np.linspace(0,0.5*max_w,ir_length*4))
        else:
            #calculate discrete transfer function of low-pass filter with pole at freq_bw
            w, H = sp.signal.freqs([freq_bw*(2*np.pi)], [1,freq_bw*(2*np.pi)], np.linspace(0,0.5*max_w,ir_length*4))

        #frequency vector for discrete transfer function in hz
        f = w/(2*np.pi)

        #plot frequency response of the low-pass filter
        plt.figure(dpi=800)
        plt.semilogx(1e-9*f,20*np.log10(abs(H)))
        plt.ylabel('Mag. Response [dB]')
        plt.xlabel('Frequency [GHz]')
        plt.title("Low Pass Filter with {}MHz Cutoff Magnitude Bode Plot".format(round(freq_bw*1e-6)))
        plt.grid()
        plt.axvline(x=1e-9*freq_bw,color = 'grey')
        plt.show()

        #find impluse response of low-pass filter
        h, t = sdp.freq2impulse(H,f)

        #plot impulse response of the low-pass filter 
        # plt.figure(dpi=800)
        # plt.plot(t[:ir_length]*1e12,h[:ir_length])
        # plt.title("Low Pass Filter with {}MHz Cutoff Impulse Response".format(round(freq_bw*1e-6)))
        # plt.xlabel('Time [ps]')
        # plt.ylabel('[V]')
        # plt.show()

        self.signal = sp.signal.fftconvolve(h[:ir_length], self.signal)

    def resample(self,samples_per_symbol):
        """ resamples signal to new oversampling ratio

        Parameters
        ----------
        samples_per_symbol: int
            new number of samples per UI
        """
        #TODO: check this

        q = samples_per_symbol/self.samples_per_symbol

        #if (self.samples_per_symbol % q != 0):
        #    print(r'Must downsample UI with a divisor of {self.samples_per_symbol}')
        #    return False
        
        self.samples_per_symbol = samples_per_symbol
        
        self.signal = samplerate.resample(self.signal, q, 'zero_order_hold')
        
        
def gaussian_jitter(signal_ideal, UI,n_symbols,samples_per_symbol,stdev):
    """Generates the TX waveform from ideal, square, self.signal_ideal with jitter

    Parameters
    ----------
    signal_ideal: array
        ideal,square transmitter voltage waveform
    
    UI: float
        length of one unit interval in seconds
    
    n_symbols: int
        number of symbols in signal_ideal
        
    samples_per_symbol: int
        number of samples in signal_ideal corresponding to one UI
        
    stdev:
        standard deviation of gaussian jitter in seconds
    
    stdev_div_UI : float
        standard deviation of jitter distribution as a pct of UI    
    """

    #generate random Gaussian distributed TX jitter values
    epsilon = np.random.normal(0,stdev,n_symbols)
    
    epsilon.clip(UI)
    epsilon[0]=0

    #calculate time duration of each sample
    sample_time = UI/samples_per_symbol

    #initializes non_ideal (jitter) array
    non_ideal = np.zeros_like(signal_ideal)

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
        non_ideal[start:end]=flip*(signal_ideal[symbol_index*samples_per_symbol-samples_per_symbol]-signal_ideal[symbol_index*samples_per_symbol])
    
    #calculate TX output waveform
    signal = np.copy(signal_ideal+non_ideal)
    return signal
