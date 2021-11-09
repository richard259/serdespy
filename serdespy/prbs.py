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
    
def prbs_checker(n, prbs_seq, data):
    """Compares array with PRBS array to check bit errors

    Parameters
    ----------
    n : int
        prbs_n number
    
    prbs_seq : array
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
    #TODO: add error condition when there is an error in the first n bits
    
    test = np.copy(data[:n])
    
    start_idx = -1
    
    for i in range (prbs_seq.size):
        if np.array_equal(prbs_seq[i:i+n], test):
            start_idx = i
            break
        
    if start_idx == -1:
        print ("invalid prbs_seq or incorrect n")
        return False
    
    #print(start_idx)
    
    error_count = 0
    
    error_idx = []
    
    for i in range(data.size):
        if (data[i] != prbs_seq[(start_idx+i)%(2**n-1)]):
            error_count = error_count+1
            error_idx = error_idx + [i]
    
    return [error_count, error_idx]