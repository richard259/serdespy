import serdespy as sdp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


#%% 2-PAM 

#generate binary data
data = sdp.prbs13(1)
     
#generate Baud-Rate sampled signal from data
signal_BR = sdp.nrz_input_BR(data)
    
#data rate in Gbps
for data_rate in [50e9,100e9]:
    
    #time per bit
    UI = 1/data_rate
    
    #define oversample ratio
    samples_per_symbol = 64
    
    #timestep
    dt = UI/samples_per_symbol

    #oversampled signal
    signal_ideal = np.repeat(signal_BR, samples_per_symbol)
    
    #eye diagram of ideal signal
    sdp.simple_eye(signal_ideal, samples_per_symbol*3, 100, dt, "{}Gbps 2-PAM Signal".format(data_rate/1e9),linewidth=1.5)
    
    #cutoff frequency
    for freq_bw in [20e9, 30e9, 40e9, 50e9]:
               
        #max frequency for constructing discrete transfer function
        max_f = 1/dt
        
        #max_f in rad/s
        max_w = max_f*2*np.pi
        
        #heuristic to get a reasonable impulse response length
        ir_length = int(4/(freq_bw*dt))
        
        #calculate discrete transfer function of low-pass filter with pole at freq_bw
        w, H = sp.signal.freqs([freq_bw*(2*np.pi)], [1,freq_bw*(2*np.pi)], np.linspace(0,0.5*max_w,ir_length*4))
        
        #frequency vector for discrete transfer function in hz
        f = w/(2*np.pi)
        
        #plot frequency response of the low-pass filter
        plt.figure(dpi=800)
        plt.semilogx(1e-9*f,20*np.log10(abs(H)))
        plt.ylabel('Mag. Response [dB]')
        plt.xlabel('Frequency [GHz]')
        plt.title("Low Pass Filter with {}GHz Cutoff Magnitude Bode Plot".format(round(freq_bw*1e-9)))
        plt.grid()
        plt.axvline(x=1e-9*freq_bw,color = 'grey')
        plt.show()
        
        #find impluse response of low-pass filter
        h, t = sdp.freq2impulse(H,f)
        
        #plot impulse response of the low-pass filter 
        # plt.figure(dpi=800)
        # plt.plot(t[:ir_length]*1e12,h[:ir_length])
        # plt.title("Low Pass Filter with {}GHz Cutoff Impulse Response".format(round(freq_bw*1e-9)))
        # plt.xlabel('Time [ps]')
        # plt.ylabel('[V]')
        # plt.show()
        
        signal_filtered = sp.signal.fftconvolve(signal_ideal, h[:ir_length], mode="full")
        
        sdp.simple_eye(signal_filtered[samples_per_symbol*100:], samples_per_symbol*3, 100, UI/samples_per_symbol, "{}Gbps 2-PAM Signal with {}GHz Cutoff Filter".format(round(data_rate/1e9),round(freq_bw*1e-9)))

#%% 4-PAM

#generate binary data
data = sdp.prqs10(1)[:10000]
     
#generate Baud-Rate sampled signal from data
signal_BR = sdp.pam4_input_BR(data)
    
#data rate in Gbps
for data_rate in [50e9,100e9]:
    
    #time per 4-PAM symbol
    UI = 2/data_rate
    
    #define oversample ratio
    samples_per_symbol = 64
    
    #timestep
    dt = UI/samples_per_symbol

    #oversampled signal
    signal_ideal = np.repeat(signal_BR, samples_per_symbol)
    
    #eye diagram of ideal signal
    sdp.simple_eye(signal_ideal, samples_per_symbol*3, 100, dt, "{}Gbps 4-PAM Signal".format(data_rate/1e9),linewidth=1.5)
    
    #cutoff frequency
    for freq_bw in [20e9, 30e9, 40e9, 50e9]:
               
        #max frequency for constructing discrete transfer function
        max_f = 1/dt
        
        #max_f in rad/s
        max_w = max_f*2*np.pi
        
        #heuristic to get a reasonable impulse response length
        ir_length = int(4/(freq_bw*dt))
        
        #calculate discrete transfer function of low-pass filter with pole at freq_bw
        w, H = sp.signal.freqs([freq_bw*(2*np.pi)], [1,freq_bw*(2*np.pi)], np.linspace(0,0.5*max_w,ir_length*4))
        
        #frequency in hz
        f = w/(2*np.pi)
        
        #plot frequency response of TF
        plt.figure(dpi=800)
        plt.semilogx(1e-9*f,20*np.log10(abs(H)))
        plt.ylabel('Mag. Response [dB]')
        plt.xlabel('Frequency [GHz]')
        plt.title("Low Pass Filter with {}GHz Cutoff Magnitude Bode Plot".format(round(freq_bw*1e-9)))
        plt.grid()
        plt.axvline(x=1e-9*freq_bw,color = 'grey')
        plt.show()
        
        #find impluse response of low-pass filter
        h, t = sdp.freq2impulse(H,f)
        
        #plot impulse response of the low-pass filter 
        # plt.figure(dpi=800)
        # plt.plot(t[:ir_length]*1e12,h[:ir_length])
        # plt.title("Low Pass Filter with {}GHz Cutoff Impulse Response".format(round(freq_bw*1e-9)))
        # plt.xlabel('Time [ps]')
        # plt.ylabel('[V]')
        # plt.show()
        
        signal_filtered = sp.signal.fftconvolve(signal_ideal, h[:ir_length], mode="full")
        
        sdp.simple_eye(signal_filtered[samples_per_symbol*100:], samples_per_symbol*3, 100, UI/samples_per_symbol, "{}Gbps 4-PAM Signal with {}GHz Cutoff Filter".format(round(data_rate/1e9),round(freq_bw*1e-9)))
        