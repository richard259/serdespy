"""Differential transfer function from 4-port network touchstone file

"""

from serdespy.chmodel import *
import numpy as np
import skrf as rf


def four_port_to_diff(network, port_def):
    """Genterates PRBS31 sequence

    Parameters
    ----------
    network : skrf Network
        4 port network object
        example: 
            s4p_file = 'path/to/touchstonefile.s4p'
            network = rf.Network(s4p_file)
    
    port_def: 2*2 array
        defines TX and RX side ports of network
        example:
            port_def = np.array([[TX1_index, RX1_index],[TX2_index, RX2_index]])
            
            PORT DEFINITIONS: 
                Port   1  ----->  TX Side      G11     RX Side  <-----  Port   2
                Port   3  ----->  TX Side      G12     RX Side  <-----  Port   4
                
            port_def = np.array([[0, 1],[2, 3]])

    Returns
    -------
    H : array
        tranfer function of differential channel
    
    f : array
        frequency vector
        
    h : array
        impulse response
    
    t = array
        time vector
        
    """
    #TODO: add options for termination and source impedance
    
    #define all in-out subnetworks and ABCD params for those networks
    
    
    #thru networks
    ch1_thru = rf.subnetwork(network, [port_def[0][0], port_def[0][1]])
    ch1_thru_abcd = rf.s2a(ch1_thru.s)

    ch2_thru = rf.subnetwork(network, [port_def[1][0], port_def[1][1]])
    ch2_thru_abcd = rf.s2a(ch2_thru.s)
    
    
    #xtalk networks
    tx1_rx2 = rf.subnetwork(network, [port_def[0][0], port_def[1][1]])
    tx1_rx2_abcd = rf.s2a(tx1_rx2.s)
    
    tx2_rx1 = rf.subnetwork(network, [port_def[0][1], port_def[1][0]])
    tx2_rx1_abcd = rf.s2a(tx2_rx1.s)

    #add termination to match charictaristic impedance of networks 12 and 34
    term1 = admittance(1/ch1_thru.z0[:,0])
    ch1_thru_abcd = series(ch1_thru_abcd,term1)

    term2 = admittance(1/ch2_thru.z0[:,0])
    ch2_thru_abcd = series(ch2_thru_abcd,term2)
    
    #get discrete transfer function for subnetworks, assuming 0 source impedance
    H1_thru = 1/ch1_thru_abcd[:,0,0]
    H2_thru = 1/ch2_thru_abcd[:,0,0]
    H_tx1_rx2 = 1/tx1_rx2_abcd[:,0,0]
    H_tx2_rx1 = 1/tx2_rx1_abcd[:,0,0]

    #Get discrete transfer function of differential signal
    H = (H1_thru + H2_thru - H_tx2_rx1 - H_tx1_rx2)/2
    
    #Get frequency response of differential transfer function
    f = network.f
    h, t = freq2impulse(H,f)
      
    
    return H, f, h, t