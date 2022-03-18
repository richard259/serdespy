"""Functions for plotting eye diagram

"""
import numpy as np
import matplotlib.pyplot as plt
from .signal import *

#add to signal
def pam4_decision_fast(x,l,m,h):
    if x<l:
        return 0
    elif x<m:
        return 1
    elif x<h:
        return 2
    else:
        return 3


def nrz_stat_eye_new(signal, samples_per_symbol, n_traces, tstep, title, probs = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1], measurement_prob = 1e-7, vrange = [-2,2], n_symbols=3):
    
    window_len = samples_per_symbol*2
    
    t = np.linspace(-(tstep*(window_len-1))/2,(tstep*(window_len-1))/2, window_len)
    
    signal = shift_signal(np.copy(signal), samples_per_symbol)
    
    simple_eye(signal, window_len, n_traces, tstep, "simple")
    
    traces = np.reshape(signal[:n_traces*samples_per_symbol], (n_traces, samples_per_symbol))
    
    #print(traces.shape)
    
    thresh = np.average(traces[:,0])
    
    #sorted traces contains lists of 1 ui for each n_symbols long pattern
    sorted_traces = [[np.zeros(traces.shape),0] for _ in range(2**n_symbols)]    
    
    taps = np.zeros(n_symbols)
    
    for i in range(n_traces):
        
        taps[1:] = taps[:-1]
        
        if traces[i][0] < thresh:
            taps[0] = 0
        else:
            taps[0] = 1
            
        if i < n_symbols:
            continue
        
        pattern_idx = 0
        
        for j in range(n_symbols):
            pattern_idx = pattern_idx + (2**j)*taps[j]
        
        pattern_idx = int(pattern_idx)
        
        #print(taps, pattern_idx)
        
        sorted_traces[pattern_idx][0][int(sorted_traces[pattern_idx][1])] = traces[i-1]
        sorted_traces[pattern_idx][1] += 1
    
    for i in range(len(sorted_traces)):
        sorted_traces[i] = sorted_traces[i][0][:int(sorted_traces[i][1])]
        
        #simple_eye(sorted_traces[i].flatten(), 32, 100, tstep, "")
    
    
    params = np.zeros((samples_per_symbol,2**n_symbols,3))
    
    for i in range(2**n_symbols):
        params[:,i,0] = np.mean(sorted_traces[i],0)
        params[:,i,1] = np.std(sorted_traces[i],0)
        params[:,i,2] = sorted_traces[i].shape[0]/n_traces
    

    cols = ["blue","cyan",(0,0.89,0.32),"yellow",'orange','red',(0.5,0,0),(0.4,0,0),(0.3,0,0)]
    
    plt.figure(dpi=1200)
    
    for j in range(len(probs)):
    
        lines = np.empty((samples_per_symbol,),dtype = object)
        
        for i in range(samples_per_symbol):
            lines[i] = find_prob(probs[j],vrange,2000,params[i,:,0],params[i,:,1],params[i,:,2])
            
        #measure eye height and eye width
        if probs[j] == measurement_prob:

            eye_width = 0
            eye_height = 0
            
            if len(lines[0]) != 4:
                print("Eye Not Open with probability", measurement_prob)
            else:
                measure = np.hstack((lines, lines))
                eye_height =  abs(measure[0][2]-measure[0][1])
                width_ctr = 0
            
                for i in range(2*samples_per_symbol):
                    if len(measure[i]) == 4:
                        width_ctr += 1     
                        if width_ctr > eye_width:
                            eye_width = width_ctr
                    else:
                        width_ctr = 0

        to_plot = lines_to_plot(lines,2**n_symbols)
        
        for i in range(0,2**n_symbols*2,2):
            if i == 0:
                plt.fill_between(t*1e12, to_plot[:,i], to_plot[:,i+1],color=cols[j], alpha=1, label = f'pdf > {probs[j]}')
            else:
                plt.fill_between(t*1e12, to_plot[:,i], to_plot[:,i+1],color=cols[j], alpha=1)
        
    plt.legend(prop={'size': 6})
    plt.title(title)
    plt.xlabel('[ps]')
    plt.ylabel('[V]')
    
    print("Eye Height: ", eye_height, "V")
    print("Eye Width: ", eye_width*tstep*1e12,"ps")
    
    return lines, to_plot

def pam4_stat_eye_new(signal, samples_per_symbol, n_traces, tstep, title, probs = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1], measurement_prob = 1e-7, vrange = [-4,4], n_symbols=3):   
    
    window_len = samples_per_symbol*2
    
    t = np.linspace(-(tstep*(window_len-1))/2,(tstep*(window_len-1))/2, window_len)
    
    signal = shift_signal(np.copy(signal), samples_per_symbol)
    
    simple_eye(signal, window_len, n_traces, tstep, "simple")
    
    traces = np.reshape(signal[:n_traces*samples_per_symbol], (n_traces, samples_per_symbol))
    
    #print(traces.shape)
    
    thresh = np.average(traces[:,0])
    
    #sorted traces contains lists of 1 ui for each n_symbols long pattern
    sorted_traces = [[np.zeros(traces.shape),0] for _ in range(4**n_symbols)]    
    
    taps = np.zeros(n_symbols)
    
    mid_thresh = np.average(traces[:,0])
    high_thresh = mid_thresh + 0.5 * np.max(traces[:,0])
    low_thresh = mid_thresh + 0.5 * np.min(traces[:,0])
    
    for i in range(n_traces):
                
        #decide on value of current bit 
        taps[1:] = taps[:-1]
        
        taps[0] = pam4_decision_fast(traces[i][0], low_thresh, mid_thresh, high_thresh)
            
        if i < n_symbols:
            continue
        
        pattern_idx = 0
        
        for j in range(n_symbols):
            pattern_idx = pattern_idx + (4**j)*taps[j]
        
        pattern_idx = int(pattern_idx)
        
        #print(taps, pattern_idx)
        
        sorted_traces[pattern_idx][0][int(sorted_traces[pattern_idx][1])] = traces[i-1]
        sorted_traces[pattern_idx][1] += 1
    
    for i in range(len(sorted_traces)):
        sorted_traces[i] = sorted_traces[i][0][:int(sorted_traces[i][1])]
        
        #simple_eye(sorted_traces[i].flatten(), 32, 100, tstep, "00")
    
    #return sorted_traces


    params = np.zeros((samples_per_symbol,4**n_symbols,3))
    
    for i in range(len(sorted_traces)):
        params[:,i,0] = np.mean(sorted_traces[i],0)
        params[:,i,1] = np.std(sorted_traces[i],0)
        params[:,i,2] = sorted_traces[i].shape[0]/n_traces
    

    cols = ["blue","cyan",(0,0.89,0.32),"yellow",'orange','red',(0.5,0,0),(0.4,0,0),(0.3,0,0)]
    
    plt.figure(dpi=1200)
    
    for j in range(len(probs)):
    
        lines = np.empty((samples_per_symbol,),dtype = object)
        
        for i in range(samples_per_symbol):
            lines[i] = find_prob(probs[j],vrange,2000,params[i,:,0],params[i,:,1],params[i,:,2])
            
        #measure eye height and eye width
        if probs[j] == measurement_prob:

            eye_width = 0
            eye_height = 0
            
            if len(lines[0]) != 8:
                print("Eye Not Open with probability", measurement_prob)
            else:
                measure = np.hstack((lines, lines))
                eye_height =  abs(measure[0][2]-measure[0][1])
                width_ctr = 0
            
                for i in range(2*samples_per_symbol):
                    if len(measure[i]) == 8:
                        width_ctr += 1     
                        if width_ctr > eye_width:
                            eye_width = width_ctr
                    else:
                        width_ctr = 0

        to_plot = lines_to_plot(lines,4**n_symbols)
        
        for i in range(0,4**n_symbols*2,2):
            if i == 0:
                plt.fill_between(t*1e12, to_plot[:,i], to_plot[:,i+1],color=cols[j], alpha=1, label = f'pdf > {probs[j]}')
            else:
                plt.fill_between(t*1e12, to_plot[:,i], to_plot[:,i+1],color=cols[j], alpha=1)
        
    plt.legend(prop={'size': 6})
    plt.title(title)
    plt.xlabel('[ps]')
    plt.ylabel('[V]')
    
    print("Eye Height: ", eye_height, "V")
    print("Eye Width: ", eye_width*tstep*1e12,"ps")
    
    return lines, to_plot


def lines_to_plot(lines, n_gaussians):
    
    to_plot = np.zeros((len(lines),2*n_gaussians))

    for i in range(len(lines)):
        #
       # print("len_lines[i]",len(lines[i]))
        
       # print("len to_print", 2*n_gaussians)
        
        
        
        factor = int((2*n_gaussians)//len(lines[i]))
        
        
       # print("factor", factor)
        
        
        for pair in range(int(len(lines[i])/2)):
            for multiple in range(factor):
                #print("to_print", (pair*factor*2+multiple*2),(pair*factor*2+multiple*2+1), "gets", pair*2,pair*2+1)
                if pair*2 >= len(lines[i]):
                    #print("to_print",pair, (pair*factor*2+multiple*2),(pair*factor*2+multiple*2+1))
                    to_plot[i, (pair*factor*2+multiple*2)] = lines[i][-1]
                    to_plot[i, (pair*factor*2+multiple*2+1)] = lines[i][-1]
                else:
                    to_plot[i, (pair*factor*2+multiple*2)] =lines[i][pair*2]
                    to_plot[i, (pair*factor*2+multiple*2)+1] =lines[i][pair*2+1]
                    #to_plot[i, (pair*factor*2+multiple*2):(pair*factor*2+multiple*2+1)] = lines[i][pair*2:pair*2+1]
            
        
        #plot = np.tile(lines[i],n_gaussians)[:2*n_gaussians]
        
        #to_plot[i,:] = plot
            
    to_plot = np.vstack((to_plot,to_plot))

    return to_plot

'''
def lines_to_plot_old(lines, n_gaussians):
    
    to_plot = np.zeros((len(lines),8))
    
    for i in range(len(lines)):
    
        if len(lines[i]) == 8:
                to_plot[i,:] = lines[i]
                            
            elif len(lines[i]) == 6:
                    to_plot[i,0] = lines[i][0]
                    to_plot[i,2] = lines[i][2]
                    to_plot[i,4] = lines[i][4]
                    to_plot[i,6] = lines[i][4]
                    
                    to_plot[i,1] = lines[i][1]
                    to_plot[i,3] = lines[i][3]
                    to_plot[i,5] = lines[i][5]
                    to_plot[i,7] = lines[i][5]
                    
            elif len(lines[i]) == 4:
                to_plot[i,0] = lines[i][0]
                to_plot[i,2] = lines[i][2]
                to_plot[i,4] = lines[i][0]
                to_plot[i,6] = lines[i][2]
                
                to_plot[i,1] = lines[i][1]
                to_plot[i,3] = lines[i][3]
                to_plot[i,5] = lines[i][1]
                to_plot[i,7] = lines[i][3]
        
                
            elif len(lines[i]) == 2:
                to_plot[i,0] = lines[i][0]
                to_plot[i,2] = lines[i][0]
                to_plot[i,4] = lines[i][0]
                to_plot[i,6] = lines[i][0]
                
                to_plot[i,1] = lines[i][1]
                to_plot[i,3] = lines[i][1]
                to_plot[i,5] = lines[i][1]
                to_plot[i,7] = lines[i][1]
                
            elif len(lines[i]) == 10:
                to_plot[i,0] = lines[i][0]
                to_plot[i,2] = lines[i][0]
                to_plot[i,4] = lines[i][0]
                to_plot[i,6] = lines[i][0]
                
                to_plot[i,1] = lines[i][9]
                to_plot[i,3] = lines[i][9]
                to_plot[i,5] = lines[i][9]
                to_plot[i,7] = lines[i][9]
        
    to_plot = np.vstack((to_plot,to_plot))
'''

def nrz_stat_eye(signal, samples_per_symbol, n_traces, tstep, title, probs = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1], measurement_prob = 1e-7, vrange = [-2,2]):
    
    
    window_len = samples_per_symbol*2
    
    t = np.linspace(-(tstep*(window_len-1))/2,(tstep*(window_len-1))/2, window_len)
    
    signal = shift_signal(np.copy(signal), samples_per_symbol)
    
    simple_eye(signal, window_len, n_traces, tstep, "simple")
    
    traces = np.reshape(signal[:n_traces*samples_per_symbol], (n_traces, samples_per_symbol))
    
    #print(traces.shape)
    
    thresh = np.average(traces[:,0])
    
    trace_00 = np.zeros(traces.shape)
    cnt_00 = 0
    
    trace_01 = np.zeros(traces.shape)
    cnt_01 = 0
    
   
    trace_10 = np.zeros(traces.shape)
    cnt_10 = 0
    
    trace_11 = np.zeros(traces.shape)
    cnt_11 = 0
    

    
    
    
    for i in range(n_traces):
        if traces[i][0] < thresh:
            if traces[i][-1] < thresh:
                trace_00[cnt_00][:] = traces[i][:]
                cnt_00 += 1
                
            else:
                trace_01[cnt_01][:] = traces[i][:]
                cnt_01 += 1
                
               
        else:
            if traces[i][-1] < thresh:
                trace_10[cnt_10][:] = traces[i][:]
                cnt_10 += 1
       
            else:
                trace_11[cnt_11][:] = traces[i][:]
                cnt_11 += 1
            
    
    trace_00 = trace_00[:cnt_00][:]
    trace_01 = trace_01[:cnt_01][:]
    trace_10 = trace_10[:cnt_10][:]
    trace_11 = trace_11[:cnt_11][:]


    simple_eye(trace_00.flatten(), 32, 200, tstep, "00")
    simple_eye(trace_01.flatten(), 32, 200, tstep, "01")
    simple_eye(trace_10.flatten(), 32, 200, tstep, "10")
    simple_eye(trace_11.flatten(), 32, 200, tstep, "11")
    
    
    params = np.zeros((samples_per_symbol,4,3))

    #print(params[:,0,0],np.mean(trace_00,0))

    params[:,0,0] = np.mean(trace_00,0)
    params[:,0,1] = np.std(trace_00,0)
    params[:,0,2] = trace_00.shape[0]/n_traces
    
    params[:,1,0] = np.mean(trace_01,0)
    params[:,1,1] = np.std(trace_01,0)
    params[:,1,2] = trace_01.shape[0]/n_traces
    
    params[:,2,0] = np.mean(trace_10,0)
    params[:,2,1] = np.std(trace_10,0)
    params[:,2,2] = trace_10.shape[0]/n_traces
    
    params[:,3,0] = np.mean(trace_11,0)
    params[:,3,1] = np.std(trace_11,0)
    params[:,3,2] = trace_11.shape[0]/n_traces
      
    cols = ["blue","cyan",(0,0.89,0.32),"yellow",'orange','red',(0.5,0,0)]
    
    plt.figure(dpi=1200)
    
    for j in range(len(probs)):
    
        lines = np.empty((samples_per_symbol,),dtype = object)
        
        for i in range(samples_per_symbol):
            lines[i] = find_prob(probs[j],vrange,2000,params[i,:,0],params[i,:,1],params[i,:,2])
            
        #measure eye height and eye width
        if probs[j] == measurement_prob:

            eye_width = 0
            eye_height = 0
            
            if len(lines[0]) != 4:
                print("Eye Not Open with probability", measurement_prob)
            else:
                measure = np.hstack((lines, lines))
                eye_height =  abs(measure[0][2]-measure[0][1])
                width_ctr = 0
            
                for i in range(samples_per_symbol):
                    if len(measure[i]) == 4:
                        width_ctr += 1     
                        if width_ctr > eye_width:
                            eye_width = width_ctr
                    else:
                        width_ctr = 0
            
        to_plot = np.zeros((samples_per_symbol,8))
        
        for i in range(samples_per_symbol):
            
            if len(lines[i]) == 8:
                to_plot[i,:] = lines[i]
    
            elif len(lines[i]) == 6:
                to_plot[i,0] = lines[i][0]
                to_plot[i,2] = lines[i][2]
                to_plot[i,4] = lines[i][4]
                to_plot[i,6] = lines[i][4]
                
                to_plot[i,1] = lines[i][1]
                to_plot[i,3] = lines[i][3]
                to_plot[i,5] = lines[i][5]
                to_plot[i,7] = lines[i][5]
                    
            elif len(lines[i]) == 4:
                to_plot[i,0] = lines[i][0]
                to_plot[i,2] = lines[i][2]
                to_plot[i,4] = lines[i][0]
                to_plot[i,6] = lines[i][2]
                
                to_plot[i,1] = lines[i][1]
                to_plot[i,3] = lines[i][3]
                to_plot[i,5] = lines[i][1]
                to_plot[i,7] = lines[i][3]

                
            elif len(lines[i]) == 2:
                to_plot[i,0] = lines[i][0]
                to_plot[i,2] = lines[i][0]
                to_plot[i,4] = lines[i][0]
                to_plot[i,6] = lines[i][0]
                
                to_plot[i,1] = lines[i][1]
                to_plot[i,3] = lines[i][1]
                to_plot[i,5] = lines[i][1]
                to_plot[i,7] = lines[i][1]
                
        
        
        to_plot = np.vstack((to_plot,to_plot))
        
             
        plt.fill_between(t*1e12, to_plot[:,0], to_plot[:,1],color=cols[j], alpha=1, label = f'p > {probs[j]}')
        plt.fill_between(t*1e12, to_plot[:,2], to_plot[:,3],color=cols[j], alpha=1)
        plt.fill_between(t*1e12, to_plot[:,4], to_plot[:,5],color=cols[j], alpha=1)
        plt.fill_between(t*1e12, to_plot[:,6], to_plot[:,7],color=cols[j], alpha=1)
        
    plt.legend(prop={'size': 6})
    plt.title(title)
    plt.xlabel('[ps]')
    plt.ylabel('[V]')
    
    print("Eye Height: ", eye_height, "V")
    print("Eye Width: ", eye_width*tstep*1e12,"ps")
    
    return True


def pam4_stat_eye(signal, samples_per_symbol, n_traces, tstep, title, probs = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1], measurement_prob = 1e-7, vrange = [-4,4]):
    
    
    window_len = samples_per_symbol*2
    t = np.linspace(-(tstep*(window_len-1))/2,(tstep*(window_len-1))/2, window_len)
    
    signal = shift_signal(np.copy(signal), samples_per_symbol)
    
    traces = np.reshape(signal[:n_traces*samples_per_symbol], (n_traces, samples_per_symbol))
    
    #print(traces.shape)
    
    mid_thresh = np.average(traces[:,0])
    high_thresh = mid_thresh + 0.5 * np.max(traces[:,0])
    low_thresh = mid_thresh + 0.5 * np.min(traces[:,0])
    
    #print(mid_thresh,high_thresh,low_thresh)
    
    
    trace_00 = np.zeros(traces.shape)
    cnt_00 = 0
    
    trace_01 = np.zeros(traces.shape)
    cnt_01 = 0
    
    trace_02 = np.zeros(traces.shape)
    cnt_02 = 0
    
    trace_03 = np.zeros(traces.shape)
    cnt_03 = 0
    
    
    trace_10 = np.zeros(traces.shape)
    cnt_10 = 0
    
    trace_11 = np.zeros(traces.shape)
    cnt_11 = 0
    
    trace_12 = np.zeros(traces.shape)
    cnt_12 = 0
    
    trace_13 = np.zeros(traces.shape)
    cnt_13 = 0
    
    
    trace_20 = np.zeros(traces.shape)
    cnt_20 = 0
    
    trace_21 = np.zeros(traces.shape)
    cnt_21 = 0
    
    trace_22 = np.zeros(traces.shape)
    cnt_22 = 0
    
    trace_23 = np.zeros(traces.shape)
    cnt_23 = 0
    
    
    trace_30 = np.zeros(traces.shape)
    cnt_30 = 0
    
    trace_31 = np.zeros(traces.shape)
    cnt_31 = 0
    
    trace_32 = np.zeros(traces.shape)
    cnt_32 = 0
    
    trace_33 = np.zeros(traces.shape)
    cnt_33 = 0
    
    
    
    
    for i in range(n_traces):
        if traces[i][0] < low_thresh:
            if traces[i][-1] < low_thresh:
                trace_00[cnt_00][:] = traces[i][:]
                cnt_00 += 1
                
            elif traces[i][-1] < mid_thresh:
                trace_01[cnt_01][:] = traces[i][:]
                cnt_01 += 1
                
            elif traces[i][-1] < high_thresh:
                trace_02[cnt_02][:] = traces[i][:]
                cnt_02 += 1
                
            else:
                trace_03[cnt_03][:] = traces[i][:]
                cnt_03 += 1
                
        elif traces[i][0] < mid_thresh:
            
            if traces[i][-1] < low_thresh:
                trace_10[cnt_10][:] = traces[i][:]
                cnt_10 += 1
                
            elif traces[i][-1] < mid_thresh:
                trace_11[cnt_11][:] = traces[i][:]
                cnt_11 += 1
                
            elif traces[i][-1] < high_thresh:
                trace_12[cnt_12][:] = traces[i][:]
                cnt_12 += 1
                
            else:
                trace_13[cnt_13][:] = traces[i][:]
                cnt_13 += 1
                
                
        elif traces[i][0] < high_thresh:
            
            if traces[i][-1] < low_thresh:
                trace_20[cnt_20][:] = traces[i][:]
                cnt_20 += 1
                
            elif traces[i][-1] < mid_thresh:
                trace_21[cnt_21][:] = traces[i][:]
                cnt_21 += 1
                
            elif traces[i][-1] < high_thresh:
                trace_22[cnt_22][:] = traces[i][:]
                cnt_22 += 1
                
            else:
                trace_23[cnt_23][:] = traces[i][:]
                cnt_23 += 1
                
                
        else:
            
            if traces[i][-1] < low_thresh:
                trace_30[cnt_30][:] = traces[i][:]
                cnt_30 += 1
                
            elif traces[i][-1] < mid_thresh:
                trace_31[cnt_31][:] = traces[i][:]
                cnt_31 += 1
                
            elif traces[i][-1] < high_thresh:
                trace_32[cnt_32][:] = traces[i][:]
                cnt_32 += 1
                
            else:
                trace_33[cnt_33][:] = traces[i][:]
                cnt_33 += 1
            
    
    trace_00 = trace_00[:cnt_00][:]
    trace_01 = trace_01[:cnt_01][:]
    trace_02 = trace_02[:cnt_02][:]
    trace_03 = trace_03[:cnt_03][:]
    
    trace_10 = trace_10[:cnt_10][:]
    trace_11 = trace_11[:cnt_11][:]
    trace_12 = trace_12[:cnt_12][:]
    trace_13 = trace_13[:cnt_13][:]
    
    trace_20 = trace_20[:cnt_20][:]
    trace_21 = trace_21[:cnt_21][:]
    trace_22 = trace_22[:cnt_22][:]
    trace_23 = trace_23[:cnt_23][:]
    
    trace_30 = trace_30[:cnt_30][:]
    trace_31 = trace_31[:cnt_31][:]
    trace_32 = trace_32[:cnt_32][:]
    trace_33 = trace_33[:cnt_33][:]
    
    
    params = np.zeros((samples_per_symbol,16,3))

    #print(params[:,0,0],np.mean(trace_00,0))

    params[:,0,0] = np.mean(trace_00,0)
    params[:,0,1] = np.std(trace_00,0)
    params[:,0,2] = trace_00.shape[0]/n_traces
    
    params[:,1,0] = np.mean(trace_01,0)
    params[:,1,1] = np.std(trace_01,0)
    params[:,1,2] = trace_01.shape[0]/n_traces
    
    params[:,2,0] = np.mean(trace_02,0)
    params[:,2,1] = np.std(trace_02,0)
    params[:,2,2] = trace_02.shape[0]/n_traces
    
    params[:,3,0] = np.mean(trace_03,0)
    params[:,3,1] = np.std(trace_03,0)
    params[:,3,2] = trace_03.shape[0]/n_traces
    
    
    params[:,4,0] = np.mean(trace_10,0)
    params[:,4,1] = np.std(trace_10,0)
    params[:,4,2] = trace_10.shape[0]/n_traces
    
    params[:,5,0] = np.mean(trace_11,0)
    params[:,5,1] = np.std(trace_11,0)
    params[:,5,2] = trace_11.shape[0]/n_traces
    
    params[:,6,0] = np.mean(trace_12,0)
    params[:,6,1] = np.std(trace_12,0)
    params[:,6,2] = trace_12.shape[0]/n_traces
    
    params[:,7,0] = np.mean(trace_13,0)
    params[:,7,1] = np.std(trace_13,0)
    params[:,7,2] = trace_13.shape[0]/n_traces
    
            
    params[:,8,0] = np.mean(trace_20,0)
    params[:,8,1] = np.std(trace_20,0)
    params[:,8,2] = trace_20.shape[0]/n_traces
    
    params[:,9,0] = np.mean(trace_21,0)
    params[:,9,1] = np.std(trace_21,0)
    params[:,9,2] = trace_21.shape[0]/n_traces
    
    params[:,10,0] = np.mean(trace_22,0)
    params[:,10,1] = np.std(trace_22,0)
    params[:,10,2] = trace_22.shape[0]/n_traces
    
    params[:,11,0] = np.mean(trace_23,0)
    params[:,11,1] = np.std(trace_23,0)
    params[:,11,2] = trace_23.shape[0]/n_traces
  
    
  
    params[:,12,0] = np.mean(trace_30,0)
    params[:,12,1] = np.std(trace_30,0)
    params[:,12,2] = trace_30.shape[0]/n_traces
    
    params[:,13,0] = np.mean(trace_31,0)
    params[:,13,1] = np.std(trace_31,0)
    params[:,13,2] = trace_31.shape[0]/n_traces
    
    params[:,14,0] = np.mean(trace_32,0)
    params[:,14,1] = np.std(trace_32,0)
    params[:,14,2] = trace_32.shape[0]/n_traces
    
    params[:,15,0] = np.mean(trace_33,0)
    params[:,15,1] = np.std(trace_33,0)
    params[:,15,2] = trace_33.shape[0]/n_traces
    
    
    #prob = 1e-3
    
    #probs = [0.00001,0.005,0.01,0.05,0.1]
    
    #probs = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,2e-1]
    
    cols = ["blue","cyan",(0,0.89,0.32),"yellow",'orange','red',(0.5,0,0)]
    
    plt.figure(dpi=1200)
    
    for j in range(len(probs)):
    
        lines = np.empty((samples_per_symbol,),dtype = object)
        
        for i in range(samples_per_symbol):
            lines[i] = find_prob(probs[j],vrange,2000,params[i,:,0],params[i,:,1],params[i,:,2])
            
        #measure eye height and eye width
        if probs[j] == measurement_prob:
            
            eye_width = 0
            eye_height = 0
            
            if len(lines[0]) != 8:
                print("Eye Not Open with probability", measurement_prob)
            else:
                measure = np.hstack((lines, lines))
                eye_height =  min(abs(measure[0][2]-measure[0][1]),abs(measure[0][3]-measure[0][4]),abs(measure[0][5]-measure[0][6]))
                width_ctr = 0
            
            
                for i in range(samples_per_symbol):
                    if len(measure[i]) == 8:
                        width_ctr += 1     
                        if width_ctr > eye_width:
                            eye_width = width_ctr
                    else:
                        width_ctr = 0

                
                
            
        to_plot = np.zeros((samples_per_symbol,8))
        
        for i in range(samples_per_symbol):
            
            if len(lines[i]) == 8:
                to_plot[i,:] = lines[i]
                                
            elif len(lines[i]) == 6:
                    to_plot[i,0] = lines[i][0]
                    to_plot[i,2] = lines[i][2]
                    to_plot[i,4] = lines[i][4]
                    to_plot[i,6] = lines[i][4]
                    
                    to_plot[i,1] = lines[i][1]
                    to_plot[i,3] = lines[i][3]
                    to_plot[i,5] = lines[i][5]
                    to_plot[i,7] = lines[i][5]
                    
            elif len(lines[i]) == 4:
                to_plot[i,0] = lines[i][0]
                to_plot[i,2] = lines[i][2]
                to_plot[i,4] = lines[i][0]
                to_plot[i,6] = lines[i][2]
                
                to_plot[i,1] = lines[i][1]
                to_plot[i,3] = lines[i][3]
                to_plot[i,5] = lines[i][1]
                to_plot[i,7] = lines[i][3]

                
            elif len(lines[i]) == 2:
                to_plot[i,0] = lines[i][0]
                to_plot[i,2] = lines[i][0]
                to_plot[i,4] = lines[i][0]
                to_plot[i,6] = lines[i][0]
                
                to_plot[i,1] = lines[i][1]
                to_plot[i,3] = lines[i][1]
                to_plot[i,5] = lines[i][1]
                to_plot[i,7] = lines[i][1]
                
            elif len(lines[i]) == 10:
                to_plot[i,0] = lines[i][0]
                to_plot[i,2] = lines[i][0]
                to_plot[i,4] = lines[i][0]
                to_plot[i,6] = lines[i][0]
                
                to_plot[i,1] = lines[i][9]
                to_plot[i,3] = lines[i][9]
                to_plot[i,5] = lines[i][9]
                to_plot[i,7] = lines[i][9]
                
        
        
        to_plot = np.vstack((to_plot,to_plot))
        
             
        plt.fill_between(t*1e12, to_plot[:,0], to_plot[:,1],color=cols[j], alpha=1, label = f'p > {probs[j]}')
        plt.fill_between(t*1e12, to_plot[:,2], to_plot[:,3],color=cols[j], alpha=1)
        plt.fill_between(t*1e12, to_plot[:,4], to_plot[:,5],color=cols[j], alpha=1)
        plt.fill_between(t*1e12, to_plot[:,6], to_plot[:,7],color=cols[j], alpha=1)
        
    plt.legend(prop={'size': 6})
    plt.title(title)
    plt.xlabel('[ps]')
    plt.ylabel('[V]')
    
    print("Eye Height: ", eye_height, "V")
    print("Eye Width: ", eye_width*tstep*1e12,"ps")
    
    return True
    



def find_prob(prob,vrange, n_points, means, stds, scales):
    v = np.linspace(vrange[0],vrange[1],n_points)
    
    below = True
    
    pts = []
    
    if multiple_gaussians(means,stds,scales,vrange[0]) > prob:
        print("gaussians out of range")
        return False
    
    for x in v:
        p = multiple_gaussians(means,stds,scales,x)
        if below:
            if p>prob:
                pts = pts + [x]
                below = False
        else:
            if p<prob:
                pts = pts + [x]
                below = True
    
    return pts

def gaussian(mean,std,scale,x):
    
    coef = 1/(std*np.sqrt(2*np.pi))
    arg = (-0.5*(x-mean)**2) / (std**2)
    return scale*coef*np.exp(arg)

def multiple_gaussians(means,stds,scales,x):
    y = 0
    for i in range(means.size):
        y = y + gaussian(means[i],stds[i],scales[i],x)

    return y

def simple_eye(signal, window_len, ntraces, tstep, title, res=1200):
    """Genterates simple eye diagram

    Parameters
    ----------
    signal: array
        signal to be plotted
    
    window_len: int
        number of time steps in eye diagram x axis
    
    ntraces: int
        number of traces to be plotted
    
    tstep: float
        timestep of time domain signal
    
    title: 
        title of the plot
    """

    signal_crop = signal[0:ntraces*window_len]
    traces = np.split(signal_crop,ntraces)

    t = np.linspace(-(tstep*(window_len-1))/2,(tstep*(window_len-1))/2, window_len)
    
    plt.figure(dpi=res)
    for i in range(ntraces):
        plt.plot(t*1e12,np.reshape((traces[i][:]),window_len), color = 'blue', linewidth = 0.15)
        plt.title(title)
        plt.xlabel('[ps]')
        plt.ylabel('[V]')
        #plt.ylim([-0.6,0.6])
    
    return True

