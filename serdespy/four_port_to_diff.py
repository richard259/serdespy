"""Differential transfer function from 4-port network touchstone file

"""

from serdespy.chmodel import *
import numpy as np
import skrf as rf


def four_port_to_diff(network, port_def, source, load, option = 0, t_d = None):
    """

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
    s_params = network.s
    f = network.f
    pts = f.size
    
    
    #change port def
    #ports = np.array([1,3,2,4])
    s_params_new = np.copy(s_params)
    
    s_params_new[:,1,:] = np.copy(s_params[:,2,:])
    s_params_new[:,2,:] = np.copy(s_params[:,1,:])
    
    s_params_new[:,:,1] = np.copy(s_params[:,:,2])
    s_params_new[:,:,2] = np.copy(s_params[:,:,1])
    
    s_params_new[:,1,2] = np.copy(s_params[:,1,2])
    s_params_new[:,2,1] = np.copy(s_params[:,2,1])
    
    s_params_new[:,1,1] = np.copy(s_params[:,2,2])
    s_params_new[:,2,2] = np.copy(s_params[:,1,1])
    
    
    #
    M = np.array([[1,-1,0,0],[0,0,1,-1],[1,1,0,0],[0,0,1,1]])
    invM = np.transpose(M)
    
    smm_params = np.zeros((4,4,pts), dtype = complex)
    
    for i in range(pts):
        smm_params[:,:,i] = (M@s_params_new[i,:,:]@invM)/2
    
    diff_s_params = smm_params[0:2,0:2,:]
    
    zl = load*np.ones((1,1,pts))
    zs = source*np.ones((1,1,pts))
    z0 = network.z0[0,0]*np.ones((1,1,pts))

    #reflection coefficients
    gammaL = (zl - z0) / (zl + z0)
    gammaL[zl == np.inf] = 1 
    
    gammaS = (zs - z0) / (zs + z0)
    gammaS[zs == np.inf] = 1
    
    gammaIn = (diff_s_params[0,0,:] + diff_s_params[0,1,:] * diff_s_params[1,0,:] * gammaL) / (1 - diff_s_params[1,1,:] * gammaL)
    
    H = diff_s_params[1,0,:] * (1 + gammaL) * (1 - gammaS) / (1 - diff_s_params[1,1,:] * gammaL) / (1 - gammaIn * gammaS) / 2
    
    H = H.reshape(pts,)

    if option == 1:
        H = H/H[0]
    
    if t_d != None:
        H, f, h, t = zero_pad(H,f,t_d)
    else:
        h, t = freq2impulse(H,f)
        
    return H, f, h, t