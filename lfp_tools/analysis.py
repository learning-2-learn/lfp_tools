from lfp_tools import general
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
from matplotlib.widgets import Slider


def plot_slider(sigs, vlines=[], markers=[], num_sigs=10, offset=['auto', 0], xrange=['auto', 0]):
    """
    Plots multiple signals with sliding window.
    Don't forget to use the command "%matplotlib notebook" first!
    Doesn't work in python lab. Switch to tree and try again
    
    Parameters
    ----------
    sigs : array (or list) of arrays to plot
    vlines : list of black, vertical lines to plot
    markers : list of specific vertical lines to plot per signal
        Should be of shape (len(sigs), any number, 4), where the last argument gives idx, color, marker, marker_size
    num_sigs : number of arrays to show in the screen at any one time
    offset : list of two objects. First (string) describes how to correct offset by second (number) object:
        'auto' automatically finds offset
        'rel' allows user to adjust automatic offset by multiplicative factor (second arg)
        'abs' allows user to set absolute offset (second arg)
    xrange : list of two objects. First (string) describes how to correct xrange by second (number) object:
        'auto' automatically finds xrange
        'rel' allows user to adjust automatic xrange by multiplicative factor (second arg)
        'abs' allows user to set absolute xrange (second arg)
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    minRange = 0
    if (xrange[0]=='auto'):
        maxRange = int(len(sigs[0]) / 100)
    elif (xrange[0]=='abs'):
        maxRange = xrange[1]
    elif (xrange[0]=='rel'):
        maxRange = int(len(sigs[0]) / 100) * xrange[1]
    else:
        print('Incorrect xrange args')
        maxRange = int(len(sigs[0]) / 100)
        
    if (offset[0]=='auto'):
        offset = 2 * np.std(sigs)
    elif (offset[0]=='rel'):
        offset = 2 * np.std(sigs) * offset[1]
    elif (offset[0]=='abs'):
        offset = offset[1]
    else:
        print('Incorrect offset args')
        offset = 2 * np.std(sigs)
        
    numPlots = len(sigs)

    plt.yticks([], [])

    for i in range(numPlots):
        plt.plot(np.arange(len(sigs[i])), sigs[i] + offset*i)
        if (markers != []):
            for m in markers[i]:
                plt.plot(m[0], offset*i, color=m[1], marker=m[2], markersize = m[3])
        
    for v in vlines:
        plt.axvline(v, color='k')

    plt.axis([minRange, maxRange, -offset, (num_sigs-1)*offset])

    aypos = plt.axes([0.01, 0.25, 0.03, 0.60])
    axpos = plt.axes([0.22, 0.1, 0.6, 0.04])

    yspos = Slider(aypos, 'Y', -offset, (numPlots-1)*offset, orientation='vertical')
    xspos = Slider(axpos, 'X', 0, len(sigs[0]) - maxRange, orientation='horizontal')

    def update(val):
        ypos = yspos.val
        xpos = xspos.val
        ax.axis([xpos,xpos + maxRange - minRange,ypos,ypos+num_sigs*offset])
        fig.canvas.draw_idle()

    yspos.on_changed(update)
    xspos.on_changed(update)

    plt.show()
    return(xspos, yspos)

def moving_average_dim(ar, size, dim):
    """
    Calculates the moving average along dimension
    
    Parameters
    --------------
    ar: array to be averaged
    size: size of window
    dim: dimension to calculate over
    
    Returns
    --------------
    Moving average along dim
    """
    br = np.apply_along_axis(_moving_average, dim, ar, size)
    return(br)


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

def _moving_average(a, n):
    """
    Calculates the moving average of an array.
    Function taken from Jaime here:
    https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    
    Parameters
    --------------
    a: array to be averaged
    n: size of window
    
    Returns
    --------------
    Moving average
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n