from .signal import *
from .chmodel import *
import numpy as np
import skrf as rf
import scipy as sp
import matplotlib.pyplot as plt
import samplerate


class Transmitter:
    
    """Class to represent time domain signal at transmitter


    """
    
    def __init__(self, data, voltage_levels, frequency):
        
        """
        Initialize transmitter, stores data and converts to array of symbols
        
        Parameters
        ----------
        data : array
            Binary sequence containing {0,1} if NRZ
            Quaternary sequence containing {0,1,2,3} symbols if PAM-4
        
        voltage levels: array
            definition of voltages corresponding to symbols. 
        
        frequency: int
            2* symbol rate      
        
        """
        
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
            self.signal_BR = nrz_input(1,data,voltage_levels)
        elif voltage_levels.size == 4:
            #self.signal_BR = pam4_input(1,data,voltage_levels)
            self.signal_BR = pam4_input_BR(data,voltage_levels)
        else:
            print ("Error: Voltage levels must have either size = 2 for NRZ signal or size = 4 for PAM4")
    
    def FIR(self, tap_weights):
        
        """Implements TX - FIR and creates  self.signal_FIR_BR = filtered, baud-rate sampled signal
        
        Parameters
        ----------
        
        tap_weights: list
            
        
        """
        self.FIR_enable = True
        
        self.signal_FIR_BR = sp.signal.fftconvolve(self.signal_BR,tap_weights, mode="same")

    def oversample(self, samples_per_symbol):
        
        """Oversamples the baud-rate-sampled signal

        Parameters
        ----------
        
        samples_per_symbol : int
            number of samples per symbol
            
        
        """
        self.samples_per_symbol = samples_per_symbol
        
        #if we have FIR filtered data
        if self.FIR_enable:
            oversampled = np.zeros(len(self.signal_FIR_BR)*self.samples_per_symbol)
            for i in range(self.n_symbols):
                oversampled[i*self.samples_per_symbol:(i+1)*self.samples_per_symbol]=self.signal_FIR_BR[i]
        
        #if we are not using FIR
        else:
            oversampled = np.zeros(len(self.signal_BR)*self.samples_per_symbol)
            for i in range(self.n_symbols):
                oversampled[i*self.samples_per_symbol:(i+1)*self.samples_per_symbol]=self.signal_BR[i]
        
        self.signal_ideal = oversampled

    def new_oversample(self, samples_per_symbol):
        #approach using samplerate
        self.samples_per_symbol = samples_per_symbol
        
        #if we have FIR filtered data
        if self.FIR_enable:
            oversampled = samplerate.resample(self.signal_FIR_BR,samples_per_symbol,converter_type='zero_order_hold')
        
        #if we are not using FIR
        else:
            oversampled = samplerate.resample(self.signal_BR,samples_per_symbol,converter_type='zero_order_hold')
        
        self.signal_ideal = oversampled
    
    def gaussian_jitter(self, stdev_div_UI = 0.025):
        """Generates the TX waveform from ideal, square, self.signal_ideal with gaussian jitter
    
        Parameters
        ----------
        stdev_div_UI : float
            multiply this by UI to get standard deviation of gaussian jitter values applied to ideal,square transmitter waveform
    
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

    def tx_bandwidth(self,freq_bw = None, TF = None):
        """Returns the bandwidth-limited version of output signal of FIR filter
        
        If this class is called without specifying freq_bw and/or TF, the default will be used

        Parameters
        ----------
        freq_bw: float
            bandwidth frequency

        TF: list
            transfer function coefficients for bandwidth-limiting (Optio)
        """

        #freq_bw = 50e9

        #TF = ([2*np.pi*freq_bw], [1,2*np.pi*freq_bw])

        #UI = 1/(2*nyquist_f)
            
        dt = self.UI/self.samples_per_symbol

        print(f'signal timestep is {dt}')

        max_f = 1/dt

        max_w = max_f/(2*np.pi)

        ir_length = int(4/(freq_bw*dt))
            
        w, H = sp.signal.freqs([freq_bw/(2*np.pi)], [1,freq_bw/(2*np.pi)], np.linspace(0,0.5*max_w,ir_length*4))

        f = np.pi*2*w

        plt.figure()
        plt.semilogx(1e-9*f,20*np.log10(abs(H)), label = "TX BW limiting TF")
        plt.ylabel('Mag. Response [dB]')
        plt.xlabel('Frequency [GHz]')
        plt.title("TX BW Magnitude Bode Plot")
        plt.grid()
        plt.axvline(x=1e-9*freq_bw,color = 'grey')
        #plt.axhline(x=-3,color = 'grey')
        plt.show()


        h, t = freq2impulse(H,f)

        print(f'impulse response timestep is {t[1]}')
        
        plt.figure()
        plt.plot(h[:ir_length])
        plt.title("TX BW Impulse Response")
        plt.show()
        
        print("running convolution of signal with impulse response...")
        self.signal = sp.signal.fftconvolve(h[:ir_length], self.signal)


    def downsample(self,q):
        """Downsamples the input signal by a factor of q

        Parameters
        ----------
        q: int
            downsample factor
        """

        self.q = q

        interpolation_time = np.linspace(0,len(self.signal),len(self.signal)//self.q,endpoint=False)
        self.signal = np.interp(interpolation_time,np.arange(len(self.signal)),self.signal)

    def new_downsample(self,q):
        #approach using samplerate

        if (self.samples_per_symbol % q != 0):
            print(r'Must downsample UI with a divisor of {self.samples_per_symbol}')
            return False
        
        self.samples_per_symbol = int(self.samples_per_symbol/q)
        
        print("samples per UI = ", self.samples_per_symbol)
        
        self.signal_downsampled = samplerate.resample(self.signal, 1/q, 'zero_order_hold')