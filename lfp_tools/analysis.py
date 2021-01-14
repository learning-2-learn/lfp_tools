from lfp_tools import general
import numpy as np
import pandas as pd
import scipy.signal as ss


def butter_pass_filter(data, cutoff, fs, btype, order=5):
    """ 
    Butter pass filters a signal with a butter filter
    
    Parameters
    ----------
    data: the signal to filter
    cutoff: the cutoff frequency
    fs: sampling rate
    btype: either \'high\' or \'low\', determines low pass or high pass filter
    order: the order of the filter
        
    Returns
    -------
    Either high or low pass filtered data
    """
    b, a = _butter_pass(cutoff, fs, btype, order=order)
    y = ss.filtfilt(b, a, data)
    return y

def get_psd(lfp, sr):
    """
    Finds the psd for the lfp data. Uses fft.
    Double check for correctness.
    
    Parameters
    ---------------
    lfp: signal to fft
    sr: sampling rate
    
    Returns
    ---------------
    freq: the frequencies associated with the power
    power: the power at each frequency
    """
    power = np.abs(np.fft.fft(lfp))
    power = power[:int(len(power)/2)]
    freq = np.arange(len(power)) * sr / (2 * len(power))
    return(freq, power)

def time_Slicer(ar, timePoints, timeLength):
    """ 
    Selects desired epochs with desired lengths from general array
    
    Parameters
    ----------
    ar: array to be sliced
    timePoints: list of beginnings of epochs
    timeLength: length of epochs
        
    Returns
    -------
    ar[idx]: 2 dimensional array of shape (len(timPoints), timeLength).
        Includes time array for each epoch
    """
    idx = timePoints[:,np.newaxis] + np.arange(0,timeLength)
    return(ar[idx])

def beh_get_breaks(df, num_std=5):
    """
    Finds the trials where a break occurs.
    
    Parameters
    ---------------
    df: behavioral dataframe
    num_std: number of standard deviations above which is considered a break
    
    Returns
    ---------------
    c: an array of what trial a break occurs
    """
    a = df[df['encode']==150].time.values
    a2 = df[df['encode']==151].time.values
    b = a[1:] - a2[:-1]
    c = np.argwhere(np.mean(b) + num_std * np.std(b) < b)[:,0]
    if (c.size > 0):
        return (c + 0.5)
    else:
        return (c)
    
def get_chan_neighbors(chan):
    """
    Finds the nearest neighbor channels to a given channel.
    
    Parameters
    --------------
    chan: input channel to find neighbors
    
    Returns
    --------------
    list of locations of nearest neighbors
    """
    channels = general.load_json_file('channels.json')
    if ('a' in chan):
        drive_chans = np.array(channels['drive_pfc'])
    else:
        drive_chans = np.array(channels['drive_temp'])
    if (chan not in np.hstack(drive_chans)):
        print('Channel name not in drive, check name.')
        return []
    elif (chan == '0'):
        print('Channel name needs to be positive')
        return []
    loc = np.argwhere(drive_chans == chan)
    ch_nn = [
        drive_chans[loc[0,0]+1, loc[0,1]],
        drive_chans[loc[0,0]-1, loc[0,1]],
        drive_chans[loc[0,0], loc[0,1]+1],
        drive_chans[loc[0,0], loc[0,1]-1]]
    return [ch for ch in ch_nn if ch != '0' and ch != '0a']

def get_bad_channels(subject, session):
    """
    Finds and returns the bad channels of a given subject and session.
    
    Parameters
    ---------------
    subject: the subject's name
    session: the session id
    
    Returns
    ---------------
    List of bad channels
    """
    bad_channels = general.load_json_file('bad_channels.json')
    all_sessions = list(bad_channels.keys())
    if (subject + session in all_sessions):
        return (bad_channels[subject + session])
    else:
        print('Either bad channels haven\'t been identified or incorrect subject/session')
        return ([])
    
    
    #Helper functions
    
def _butter_pass(cutoff, fs, btype, order=5):
    """ 
    Builds a butter pass filter
    
    Parameters
    ----------
    cutoff: the cutoff frequency
    fs: sampling rate
    btype: either \'high\' or \'low\', determines low pass or high pass filter
    order: the order of the filter
        
    Returns
    -------
    Either high or low pass filtered
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = ss.butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a