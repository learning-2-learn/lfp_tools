from lfp_tools import general
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
from matplotlib.widgets import Slider


def reorder_chans(chans):
    '''
    Takes a list of channels and reorders them by number and drive.
    Specifically, it returns the idx to reorder channels
    
    Parameters
    ----------
    chans : list or array of channel names. Names must be strings of integers or integers with 'a' on the end
    
    Returns
    -------
    idx : array of indicies that sort the channels
    '''
    chans = np.array(chans)
    chans_temp = np.sort([int(c) for c in chans if 'a' not in c])
    chans_post = np.sort([int(c.split('a')[0]) for c in chans if 'a' in c])
    chans_temp = [str(c) for c in chans_temp]
    chans_post = [str(c) + 'a' for c in chans_post]
    chans_sorted = chans_temp + chans_post
    idx = np.array([list(chans).index(c) for c in chans_sorted])
    return (idx)

def get_reordered_idx(df, i_type, params=[]):
    """
    Finds idx orderings and trial changes for ordering trials in unique way.
    \'rule\' organizes by rule, in order of appearance
    \'rule_basic\' just gives trial changes for rules (doesn't reorganize)
    \'feedback\' organizes by whether the subject was correct or not, in order of appearance
    \'feedback_rule' organizes by both rule and feedback, in order
    \'time_learned' organizes by what relative trial it is
    
    Parameters
    --------------
    
    df : dataframe describing behavior
    i_type : the control, specified above
    
    Returns
    --------------
    idx : indicies for reorganizing trials
    hlines : the trial number for segregating reorganization. E.g. separating each rule
    
    """
    idx = []
    hlines = []
    if (i_type == 'feedback'):
        for i in [200,206]:
            idx.append(np.argwhere(df[df['act']=='fb'].response.values==i)[:,0])
            hlines.append(len(np.hstack(idx)) - 0.5)
        idx = np.hstack(idx)
    elif (i_type == 'feedback_previous'):
        for i in [200,206]:
            idx.append((np.argwhere(df[df['act']=='fb'].response.values==i)[:,0]+1)[:-1])
            hlines.append(len(np.hstack(idx)) - 0.5)
        idx = np.hstack(idx)
    elif (i_type == 'rule'):
        for i in range(12):
            idx.append(np.argwhere(df[df['act']=='fb'].rule.values==i)[:,0])
            hlines.append(len(np.hstack(idx)) - 0.5)
        idx = np.hstack(idx)
    elif (i_type == 'rule_basic'):
        idx = np.arange(len(df[df['act']=='fb']))
        hlines = np.argwhere(df[(df['act']=='fb')].trialRel.values==0)[1:,0] - 0.5
    elif (i_type == 'feedback_rule'):
        for j in [200,206]:
            for i in range(12):
                idx.append(np.argwhere((df[df['act']=='fb'].rule.values==i) & (df[df['act']=='fb'].response.values==j))[:,0])
                hlines.append(len(np.hstack(idx)) - 0.5)
            hlines.append(hlines[-1])
        idx = np.hstack(idx)
    elif (i_type == 'time_learned'):
        if (params!=[]):
            numTrials = params[0]
        else:
            numTrials = np.max(df[df['act']=='fb'].trialRel.values)
        for i in range(numTrials):
            idx.append(np.argwhere(df[df['act']=='fb'].trialRel.values==i)[:,0])
            hlines.append(len(np.hstack(idx)) - 0.5)
        idx = np.hstack(idx)
    else:
        print('Type not found, please use one of the following:\n \
        \'rule\', \'rule_basic\', \'feedback\', \'feedback_previous\', \'feedback_rule\', \'time_learned\'')
        return([],[])
    return(idx, np.array(hlines)[:-1])

        
def plot_grid(function, plots, grid=(3,4), figsize=(14,7), sharex=True, sharey=True, titles=[], vlines=[], vline_colors=[], hlines=[], saveFig=None, **plt_kwargs):
    """
    Possibility of adding more plotting functions...
    
    Plots a grid of individual plots with specified function
    
    Parameters
    ----------
    function : can either be a string that specifies the function or a function that produces the desired plot
    plots : list of plots, must be of length greater than gridsize
    grid : (num_vertical, num_horizontal), number of plots in grid in each direction
    figsize : (size in x direction, size in y direction)
    sharex : boolean to tell whether to share x axis
    sharey : boolean to tell whether to share y axis
    titles : list of titles for plots in same order as plots
    vlines : list of vertical lines to include
    hlines : list of horizontal lines to include
    saveFig : string of filename to save figure as. If None, does not save
    **plt_kwargs : arguments to be passed into the plotting function
    
    Functions
    ---------
    'imshow' : plots with matplotlib.pyplot.imshow
        plots elements should be in the form of an input to matplotlib.pyplot.imshow
    '_mean_and_std' : plots mean and standard deviation with matplotlib.pyplot.plot
        plots elements should have a shape of (x, ((list of arrays), (list of arrays), ...up to four times))
    '_mean_and_sample' : plots mean and spaghetti plot with matplotlib.pyplot.plot
        plots elements should have a shape of (x, ((list of arrays), (list of arrays), ...up to four times))
    """
    
    def _imshow(pl, ax=None, **plt_kwargs):
        if ax is None:
            ax = plt.gca()
        ax.imshow(pl, **plt_kwargs)
        return(ax)
    
    def _mean_and_std(pl, ax=None, **plt_kwargs):
        if ax is None:
            ax = plt.gca()
        color_mean = ['blue', 'red', 'green', 'yellow']
        color_std = ['cornflowerblue', 'lightcoral', 'lawngreen', 'lightyellow']
        x = pl[0]
        y = pl[1]
        for k in range(len(y)):
            p_mean = np.mean(y[k], axis=0)
            p_std = np.std(y[k], axis=0)
            ax.plot(x, p_mean, color=color_mean[k])
            ax.fill_between(x, p_mean+p_std, p_mean-p_std, color=color_std[k], alpha=0.5, **plt_kwargs)
        return(ax)
    
    def _mean_and_sample(pl, ax=None, **plt_kwargs):
        if ax is None:
            ax = plt.gca()
        color_mean = ['blue', 'red', 'green', 'yellow']
        color_all = ['cornflowerblue', 'lightcoral', 'lawngreen', 'lightyellow']
        num_spag = 30
        x = pl[0]
        y = pl[1]
        for k in range(len(y)):
            for spag in y[k][np.random.choice(np.array(len(y[k])), num_spag, replace=False)]:
                ax.plot(x, spag, color=color_all[k], linewidth=2, alpha=0.4)
        for k in range(len(y)):
            p_mean = np.mean(y[k], axis=0)
            ax.plot(x, p_mean, color=color_mean[k])
        return(ax)
    
    maxi=grid[0]
    maxj=grid[1]
    
    fig, ax = plt.subplots(maxi,maxj,figsize=figsize,sharex=sharex, sharey=sharey)
    fig.tight_layout(pad=1)
    
    if (vline_colors==[]):
        for i in range(len(vlines)):
            vline_colors.append('k')
    
    for i in range(maxi):
        for j in range(maxj):
            if (function=='imshow'):
                ax[i,j] = _imshow(plots[j + i*maxj], ax[i,j], **plt_kwargs)
            elif (function=='mean_and_std'):
                ax[i,j] = _mean_and_std(plots[j + i*maxj], ax[i,j], **plt_kwargs)
            elif (function=='mean_and_sample'):
                ax[i,j] = _mean_and_sample(plots[j + i*maxj], ax[i,j], **plt_kwargs)
            elif (callable(function)):
                ax[i,j] = function(plots[j + i*maxj], ax[i,j], **plt_kwargs)
            else:
                print('Bad function variable')
                return
            
            if (titles!=[]):
                ax[i,j].set_title(titles[j + i*maxj])
            for line in hlines:
                ax[i,j].axhline(line, color='black', ls='-', lw=0.7)
            for line in range(len(vlines)):
                ax[i,j].axvline(vlines[line], color=vline_colors[line], ls='-', lw=0.7)
    
    if(saveFig!=None):
        plt.savefig(saveFig, bbox_inches = 'tight')
        

def plot_slider(sigs, vlines=[], markers=[], num_sigs=10, offset=['auto', 0], xrange=['auto', 0], colors=None):
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
    colors : list of colors of each plot. Should be longer than number of plots
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
        if (colors==None):
            plt.plot(np.arange(len(sigs[i])), sigs[i] + offset*i)
        else:
            plt.plot(np.arange(len(sigs[i])), sigs[i] + offset*i, color=colors[i])
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

def get_psd(lfp, sr, params=[4096], method='welch'):
    """
    Finds the psd for the lfp data.
    Double check for correctness.
    
    Parameters
    ---------------
    lfp: signal to fft
    sr: sampling rate
    method: 'welch', 'fft', how the psd is calculated
    
    Returns
    ---------------
    freq: the frequencies associated with the power
    power: the power at each frequency
    """
    if(method=='welch'):
        freq, power = ss.welch(lfp, fs = sr, nperseg = params[0])
    elif(method=='fft'):
        power = np.abs(np.fft.fft(lfp))
        power = power[:int(len(power)/2)]
        freq = np.arange(len(power)) * sr / (2 * len(power))
    else:
        print('Wrong method, defaulting to \'welch\'')
        freq, power = get_psd(lfp, sr, params=[4096], method='welch')
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
    
def get_chan_neighbors(subject, exp, chan):
    """
    Finds the nearest neighbor channels to a given channel.
    
    Parameters
    --------------
    subject: subject selected
    exp: experiment selected
    chan: input channel to find neighbors
    
    Returns
    --------------
    list of locations of nearest neighbors
    """
    channels = general.load_json_file('sub-'+subject+'_exp-'+exp+'_channels.json')
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

def get_bad_channels(subject, exp, session):
    """
    Finds and returns the bad channels of a given subject and session.
    
    Parameters
    ---------------
    subject: the subject's name
    exp: the experiment selected
    session: the session id
    
    Returns
    ---------------
    List of bad channels
    """
    bad_channels = general.load_json_file('sub-'+subject+'_exp-'+exp+'_bad_channels.json')
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