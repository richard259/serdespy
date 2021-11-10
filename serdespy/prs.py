"""Functions for Pseudo-Random Binary Sequences

"""

import numpy as np 

def prbs31(seed):
    """Genterates PRBS31 sequence

    Parameters
    ----------
    seed : int
        seed used to generate sequence
        should be greater than 0 and less than 2^31

    Returns
    -------
    array:
        PRBS31 sequence
    """
    if (type(seed)!= int) or (seed>0x7fffffff) or (seed < 1):
        print("seed must be positive int less than 2^31")
        return False
    
    code = seed
    seq = np.zeros(2**31-1, dtype=np.uint8)
    i = 0
    sequence_complete = False
    
    while(i<2**31):
        next_bit = ((code>>30) ^ (code>>27)) & 0x00000001
        code = ((code<<1) | next_bit) & 0x7fffffff
        seq[i] = next_bit
        i = i+1
        if (i%1e6 ==0):
            print("i =", i)
        if (code==seed):
            sequence_complete = True
            break
        
    if sequence_complete:
        return seq
    else:
        print ("error, PRBS sequence did not complete")
        return False


def prbs20(seed):
    """Genterates PRBS20 sequence

    Parameters
    ----------
    seed : int
        seed used to generate sequence
        should be greater than 0 and less than 2^20

    Returns
    -------
    array:
        PRBS20 sequence
    """
    if (type(seed)!= int) or (seed>0xfffff) or (seed < 1):
        print("seed must be positive int less than 2^20")
        return False
    
    code = seed
    seq = np.zeros(2**20-1, dtype=np.uint8)
    i = 0
    sequence_complete = False
    
    while(i<2**20):
        next_bit = ((code>>19) ^ (code>>2)) & 0x00001
        code = ((code<<1) | next_bit) & 0xfffff
        seq[i] = next_bit
        i = i+1
        if (code==seed):
            sequence_complete = True
            break
        
    if sequence_complete:
        return seq
    else:
        print ("error, PRBS sequence did not complete")
        return False

def prbs13(seed):
    """Genterates PRBS13 sequence

    Parameters
    ----------
    seed : int
        seed used to generate sequence
        should be greater than 0 and less than 2^13

    Returns
    -------
    array:
        PRBS13 sequence
    """
    if (type(seed)!= int) or (seed>0x1fff) or (seed < 1):
        print("seed must be positive int less than 2^13")
        return False
    
    code = seed
    seq = np.zeros(2**13-1, dtype=np.uint8)
    i = 0
    sequence_complete = False
    
    while(i<2**20):
        next_bit = ((code>>12) ^ (code>>11) ^ (code>>1) ^ (code) ) & 0x00001
        code = ((code<<1) | next_bit) & 0x1fff
        seq[i] = next_bit
        i = i+1
        if (code==seed):
            sequence_complete = True
            break
        
    if sequence_complete:
        return seq
    else:
        print ("error, PRBS sequence did not complete")
        return False

def prbs7(seed):
    """Genterates PRBS7 sequence

    Parameters
    ----------
    seed : int
        seed used to generate sequence
        should be greater than 0 and less than 2^7

    Returns
    -------
    array:
        PRBS7 sequence
    """
    if (type(seed)!= int) or (seed>0x7f) or (seed < 1):
        print("seed must be positive int less than 2^7")
        return False
    
    code = seed
    seq = np.zeros(2**7-1, dtype=np.uint8)
    i = 0
    sequence_complete = False
    
    while(i<2**7):
        next_bit = ((code>>6) ^ (code>>5))&0x01
        code = ((code<<1) | next_bit) & 0x7f
        seq[i] = next_bit
        i = i+1
        if (code==seed):
            sequence_complete = True
            break
        
    if sequence_complete:
        return seq
    else:
        print ("error, PRBS sequence did not complete")
        return False

def prqs10(seed):
    """Genterates PRQS10 sequence

    Parameters
    ----------
    seed : int
        seed used to generate sequence
        should be greater than 0 and less than 2^20

    Returns
    -------
    array:
        PRQS10 sequence
    """
    
    a = prbs20(seed)
    shift = int((2**20-1)/3)
    b = np.hstack((a[shift:],a[:shift]))
    
    c = np.vstack((a,b))

    pqrs = np.zeros(a.size,dtype = np.uint8)
    
    for i in range(a.size):
        pqrs[i] = grey_code(c[:,i])
    
    return pqrs

def natural_code(x):
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
    
def grey_code(x):
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




def prbs_checker(n, prbs, data):
    """Compares array with PRBS array to check bit errors

    Parameters
    ----------
    n : int
        prbs_n number
    
    prbs : array
        prbs_n sequence
        
    data: array
        seqence to be checked

    Returns
    -------
    error_count : int
        number of errors in data
        
    error_idx :  list
        indexes of error bits in data
    """
    #TODO: add condition when there is an error in the first n bits
    
    test = np.copy(data[:n])
    
    start_idx = -1
    
    for i in range (prbs.size):
        if np.array_equal(prbs[i:i+n], test):
            start_idx = i
            break
        
    if start_idx == -1:
        print ("invalid prbs_seq or incorrect n")
        return False
    
    #print(start_idx)
    
    error_count = 0
    
    error_idx = []
    
    for i in range(data.size):
        if (data[i] != prbs[(start_idx+i)%(2**n-1)]):
            error_count = error_count+1
            error_idx = error_idx + [i]
    
    return [error_count, error_idx]


def prqs_checker(n, prqs, data):
    """Compares array with PRQS array to check bit errors

    Parameters
    ----------
    n : int
        prqs_n number
    
    prqs : array
        prqs_n sequence
        
    data: array
        seqence to be checked

    Returns
    -------
    error_count : int
        number of errors in data
        
    error_idx :  list
        indexes of error bits in data
    """
    #TODO: add condition when there is an error in the first n bits
    
    test = np.copy(data[:n])
    
    start_idx = -1
    
    for i in range (prqs.size):
        if np.array_equal(prqs[i:i+n], test):
            start_idx = i
            break
        
    if start_idx == -1:
        print ("invalid prqs or incorrect n")
        return False
    
    #print(start_idx)
    
    error_count = 0
    
    error_idx = []
    
    for i in range(data.size):
        if (data[i] != prqs[(start_idx+i)%(2**(n*2)-1)]):
            error_count = error_count+1
            error_idx = error_idx + [i]
    
    return [error_count, error_idx]