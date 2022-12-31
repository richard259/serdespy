"""
This file contains miscellaneous functions for digital signal processing
"""

from .chmodel import *
import numpy as np
import skrf as rf
import scipy as sp
import matplotlib.pyplot as plt

def grey_encode(x):
    """encodes 2 bits into one pam4 symbol with grey encoding

    Parameters
    ----------
    x : array
        contains the two bits to be encoded
    
    Returns
    -------
    int
        the pam4 symbol that represents x

    """
    if (x[0] == 0):
        if (x[1] == 0):
            return 0
        else:
            return 1
    else:
        if (x[1] == 0):
            return 3
        else:
            return 2
        
def grey_decode(symbol):
    """decodes one pam4 symbol into two bits with grey decoding

    Parameters
    ----------
    symbol : int
        the pam4 symbol
    
    Returns
    -------
    array
        the two bits that represent symbol
    """
    if symbol == 0 :
        return np.array([0,0],dtype = np.uint8)
    if symbol == 1 :
        return np.array([0,1],dtype = np.uint8)
    if symbol == 2 :
        return np.array([1,1],dtype = np.uint8)
    if symbol == 3 :
        return np.array([1,0],dtype = np.uint8)

def natural_encode(x):
    """encodes 2 bits into one pam4 symbol with natural encoding

    Parameters
    ----------
    x : array
        contains the two bits to be encoded
    
    Returns
    -------
    int 
        the pam4 symbol that represents x

    """
    
    if (x[0] == 0):
        if (x[1] == 0):
            return 0
        else:
            return 1
    else:
        if (x[1] == 0):
            return 2
        else:
            return 3
        
def natural_decode(symbol):
    """decodes one pam4 symbol into two bits with natural decoding

    Parameters
    ----------
    symbol : int
        the pam4 symbol
    
    Returns
    -------
    array
        the two bits that represent symbol
    """
    if symbol == 0 :
        return np.array([0,0],dtype = np.uint8)
    if symbol == 1 :
        return np.array([0,1],dtype = np.uint8)
    if symbol == 2 :
        return np.array([1,0],dtype = np.uint8)
    if symbol == 3 :
        return np.array([1,1],dtype = np.uint8)   
    
def pam4_decision(x,l,m,h):
    """produces pam4 symbol from analog voltage
    
    Parameters
    ----------
    x : float
        the analog voltage
    
    l: float
        voltage threshold between 0 and 1 symbol
    
    m: float
        voltage threshold between 1 and 2 symbol
        
    h: float
        voltage threshold between 2 and 3 symbol
    
    Returns
    -------
    int
        the pam4 symbol that represents x
    """
    if x<l:
        return 0
    elif x<m:
        return 1
    elif x<h:
        return 2
    else:
        return 3

def nrz_decision(x,t):
    """produces nrz symbol from analog voltage
    
    Parameters
    ----------
    x : float
        the analog voltage
    
    t: float
        voltage threshold between 0 and 1 symbol
    
    Returns
    -------
    int
        the nrz symbol that represents x
    """
    if x<t:
        return 0
    else:
        return 1


def pam4_input_BR(data_in, voltage_levels = np.array([-3, -1, 1, 3])):
    """produces baud-rate-sampled voltage waveform from quaternary sequence 
    
    Parameters
    ----------
    data_in : array
        quaternary sequence containing pam4 symbols
    
    voltage_levels: array
        voltage levels corresponding to each pam4 symbol
    
    Returns
    -------
    signal : array
        baud-rate-sampled voltage waveform corresponding to data_in
    """

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

def nrz_input_BR(data_in, voltage_levels = np.array([-1, 1])):
    """produces baud-rate-sampled voltage waveform from binary sequence 
    
    Parameters
    ----------
    data_in : array
        binary sequence
    
    voltage_levels: array
        voltage levels corresponding to 0 and 1 bits
    
    Returns
    -------
    signal : array
        baud-rate-sampled voltage waveform corresponding to data_in
    """

    signal = np.zeros(data_in.size)
    
    for i in range(data_in.size):
        if (data_in[i]==0):
            signal[i] = voltage_levels[0]
        elif (data_in[i]==1):
            signal[i] = voltage_levels[1]
        else:
            print('unexpected symbol in data_in')
            return False
    
    return signal

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
        voltage threshold between 0 and 1 symbol
    
    Returns
    -------
    data : array
        binary sequence generated from signal

    """
    
    data = np.zeros(int(signal.size/samples_per_symbol), dtype = np.uint8)
    
    for i in range(data.size):
        data[i] = nrz_decision(signal[i*samples_per_symbol], threshold)
            
    return data

def pam4_a2d(signal, samples_per_symbol, l, m, h):
    """slices signal and compares to voltage_levelsto convert analog signal vector to binary sequence
        signal is sampled at indeicies that are multiples of samples_per_symbol

    Parameters
    ----------
    signal: array
        signal vector at RX
        
    samples_per_symbol: int
        number of samples per UI
        
    l: float
        voltage threshold between 0 and 1 symbol
    
    m: float
        voltage threshold between 1 and 2 symbol
        
    h: float
        voltage threshold between 2 and 3 symbol
    
    Returns
    -------
    data : array
        quaternary sequence generated from signal

    """
    
    data = np.zeros(int(signal.size/samples_per_symbol), dtype = np.uint8)
    
    for i in range(data.size):
        data[i] = pam4_decision(signal[i*samples_per_symbol],l,m,h)
            
    return data


def channel_coefficients(pulse_response, t, samples_per_symbol, n_precursors, n_postcursors, res=1200, title = "Channel Coefficients"):
    #TODO: check for not long enough signal
    #make t optional arg
    """measures and plots channel coefficients from pulse response

    Parameters
    ----------
    pulse_response: array
    
    t: array
        time vector corresponding to pulse response

    samples_per_symbol: int
        number of samples per UI
        
    n_precursors : int
        number of UI before main cursor to measure
        
    n_postcursors : int
        number of UI before after cursor to measure
        
    res : int
        resolution of plot
        
    title: str
        title of plot
    
    Returns
    -------
    channel_coefficients : array
        channel coefficents measured from pulse response

    """
    #number of channel coefficients
    n_cursors = n_precursors + n_postcursors + 1
    channel_coefficients = np.zeros(n_cursors)
    
    #for plotting
    t_vec = np.zeros(n_cursors)
    xcoords = []
    half_symbol = int(round(samples_per_symbol/2))
    
    #find peak of pulse response = main cursor sample time
    max_idx = np.where(pulse_response == np.amax(pulse_response))[0][0]
    
    for cursor in range(n_cursors):
        
        #index of channel coefficint
        a = cursor - n_precursors
        
        #measure pulse response
        channel_coefficients[cursor] = pulse_response[max_idx+a*samples_per_symbol]
        
        #for plotting
        xcoords = xcoords + [1e9*t[max_idx+a*samples_per_symbol-half_symbol]]
        t_vec[a+n_precursors] = t[max_idx + a*samples_per_symbol]
        
    xcoords = xcoords + [1e9*t[max_idx+(n_postcursors+1)*samples_per_symbol-half_symbol]]
    
    
    #plot pulse response and cursor samples
    plt.figure(dpi=res)
    plt.plot(t_vec*1e9, channel_coefficients, 'o',label = 'Cursor samples')
    plt.plot(t*1e9,pulse_response, label = 'Pulse Response')
    plt.xlabel("Time [ns]")
    plt.ylabel("[V]")
    
    ll = t[max_idx-samples_per_symbol*(n_precursors+2)]*1e9
    ul = t[max_idx+samples_per_symbol*(n_postcursors+2)]*1e9
    
    #print(ll,ul)
    plt.xlim([ll,ul])
    plt.title(title)
    plt.legend()
    
    #lines to seperate UI
    for xc in xcoords:
        plt.axvline(x=xc,color = 'grey',label ='UIs', linewidth = 0.25)
    
    return channel_coefficients

def forcing_ffe(n_taps_pre, channel_coefficients, target = None):
    """generates ffe tap coefficients that force pulse response to target (zero-forcing if target not specified)

    Parameters
    ----------
    n_taps_pre:
        number of channel coefficients before main_cursor
    
    channel_coefficients: array
        channel coeffients that will be affected by FFE

    target: array
        this specifies what the pulse response will be forced to be
        should have same length as channel_coefficients
    
    Returns
    -------
    ffe_tap_weights : array
        ffe tap weights that force channel_coefficients to target

    """
    
    #number of taps
    n_taps = channel_coefficients.size

    n_taps_post = n_taps - n_taps_pre -1
    
    #build matrix for zero-forcing
    A = np.zeros((n_taps,n_taps))

    for i in range(n_taps):
        A += np.diag(np.ones(n_taps-abs(i-n_taps_pre))*channel_coefficients[i],(n_taps_pre-i) )


    if target == None:
        #do zero-forcing
        c = np.zeros((n_taps,1))
        c[n_taps_pre] = 1
    else:
        c = target

    #solve matrix inversion
    b = np.linalg.inv(A)@c

    #normalize ffe tap weights so main-cursor tap weight is 1
    b = b/np.sum(abs(b))

    ffe_tap_weights = b.T[0]

    ffe_tap_weights = ffe_tap_weights * 1/ffe_tap_weights[n_taps_pre]
    
    return ffe_tap_weights


def shift_signal(signal, samples_per_symbol):
    """Shifts signal vector so that 0th element is at centre of eye, heuristic

    Parameters
    ----------
    signal: array
        signal vector at RX
        
    samples_per_symbol: int
        number of samples per UI
    
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

def lms_equalizer(y, mu, N, w_ffe, FFE_pre, w_dfe, voltage_levels,
        alpha=None, reference=None, update_rate=1):
    """optimizes DFE and/or FFE weights for a particular signal, by using a
    Least-Mean Squares (LMS) objective function.

    Parameters
    ----------
    y: array
        signal to be equalized
    mu: number
        learning rate factor for the LMS update
    N: int
        length of `y` to be equalized
    w_ffe: array
        initial FFE weights, starting fromt the last pre-tap (oldest) to the
        last post-tap (future-most)
    FFE_pre: int
        number of pre-tap weights in FFE. The number of post-taps are implied
        based on len(w_ffe) and FFE_pre.
    w_dfe: array
        initial DFE weights, starting from the first pre-tap.
    voltage_levels: array
        discrete voltage levels to use in DFE optimization
    alpha: array (optional)
        used to fix the first `len(alpha)` weights in `w_dfe`, starting from the
        first pre-tap. these weights will not be optimized and will stay constant
        throughout LMS updates.
    reference: array (optional)
        if provided, LMS will attempt to optimize according to the error between
        the signal and reference, rather than estimating it based on the cloest
        voltage level. `len(reference)` may be less than `len(y)`, in which case
        only the first `len(reference)` updates will be optimized in this manner.
    update_rate: int (optional)
        LMS update will only be performed on each `update_rate`-th iteration.
        However, the equalized signal will still be computed on the skipped iterations.

    Returns
    -------
    w_ffe: array
        optimized FFE weights
    w_dfe: array
        optimized DFE weights
    v_ffe: array
        signal computed after FFE optimization
    v_dfe: array
        signal computed after FFE and DFE optimization
    z: array
        signal quantized from v_dfe
    e: array
        evaluated error used for LMS optimization
    """
    if w_ffe is None and w_dfe is None:
        print("warning: training nothing since both w_ffe and w_dfe were None")

    is_cooptimizing = w_ffe is not None and w_dfe is not None
    if (not is_cooptimizing) and (alpha is not None):
        print("warning: alpha is set but we're not co-optimizing")

    voltage_levels = voltage_levels.copy()

    if w_ffe is not None:
        w_ffe = np.array(w_ffe)
        FFE_taps = len(w_ffe)
        FFE_post = FFE_taps - FFE_pre - 1
    else:
        FFE_taps = 0
        FFE_post = 0

    if w_dfe is not None:
        w_dfe = np.array(w_dfe)
        DFE_taps = len(w_dfe)
    else:
        DFE_taps = 0

    e = np.zeros(N)
    v_dfe = np.zeros(N)
    v_ffe = np.zeros(N)
    z = np.zeros(N)

    min_delay = max(FFE_post, DFE_taps)
    for k in range(min_delay, N - FFE_pre):
        y_k = y[k - FFE_post:k + FFE_pre + 1][::-1]
        z_k = z[k - DFE_taps:k][::-1]

        # alpha takes over the first len(alpha) terms of w_dfe
        if is_cooptimizing and alpha is not None:
            w_dfe[:len(alpha)] = alpha

        if w_ffe is not None:
            v_ffe[k] = np.dot(y_k, w_ffe)
        else:
            v_ffe[k] = y_k[0]

        if w_dfe is not None:
            v_dfe[k] = v_ffe[k] - np.dot(z_k, w_dfe)
        else:
            v_dfe[k] = v_ffe[k]

        z[k] = _quantize(v_dfe[k], voltage_levels)
        if reference is None or k >= len(reference):
            y_ref = z[k]
        else:
            y_ref = reference[k]

        e[k] = v_dfe[k] - y_ref
        if (k - min_delay) % update_rate == 0:
            #print(k)
            if w_ffe is not None:
                w_ffe -= mu * e[k] * y_k
            if w_dfe is not None:
                w_dfe += mu * e[k] * z_k

    if is_cooptimizing and alpha is not None:
        w_dfe[:len(alpha)] = alpha

    end = N - FFE_pre
    return w_ffe, w_dfe,\
            v_ffe[min_delay:end],\
            v_dfe[min_delay:end],\
            z[min_delay:end],\
            e[min_delay:end]

def _quantize(signal, voltage_levels):
    idx = np.abs(voltage_levels - signal).argmin()
    return voltage_levels[idx]

